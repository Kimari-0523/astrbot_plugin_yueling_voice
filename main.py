import asyncio
import copy
import random
import re
import time
import uuid
import wave
from pathlib import Path

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import At, Plain, Record
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register
from astrbot.core.star.session_llm_manager import SessionServiceManager

WANT = re.compile(
    r"(?:用|发|来|说|播|读|听|要|给|换|转).{0,12}(?:语音|声音)|(?:语音|声音).{0,12}(?:回复|回答|说|读|播|来|发)|念给我听|读给我听|朗读|念出来|读出来"
)
NO_VOICE = re.compile(
    r"(?:不要|不用|别|不必|禁止|关闭|停止).{0,6}(?:语音|朗读|声音)|(?:只|仅)(?:用|发)?文字|文字(?:就好|即可|回复|回答)"
)
CONTEXT_RULE = (
    "[柚铃会话衔接] 先确定当前发言者本轮实际问的是什么，再结合当前会话已有历史、明确引用或回复的消息理解指代。"
    "语音转写和文字使用同一份会话上下文；不要把其他群友的无关发言当成本轮要求，不跨群或跨私聊拼接话题。"
    "优先回答最新问题，不重复回答已经结束的话题。指代或识别内容不清楚、现有上下文不足时，简短询问确认，不编造前情或替用户补话。"
    "切换语音只改变输出方式，文字与语音必须表达同一轮最终答复。"
)


def restore_edge_sdk_headers():
    # AstrBot 4.28.2 overwrites this shared SDK header with astrbot/<version>,
    # which the Edge service rejects with HTTP 403. Retain the SDK's own value
    # so its browser version stays consistent with its current GEC protocol.
    from edge_tts.constants import BASE_HEADERS, WSS_HEADERS

    WSS_HEADERS["User-Agent"] = BASE_HEADERS["User-Agent"]


def speech_language(text):
    """Use the answer's script, not the user's language or another session.

    Chinese sentences containing English brand names remain Chinese. Kanji-only
    text is ambiguous and falls back to Chinese; kana identify Japanese.
    """
    kana = len(re.findall(r"[\u3041-\u3096\u30a1-\u30fa\uff66-\uff9d]", text))
    han = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", text))
    if kana and kana * 2 >= han:
        return "ja"
    if han:
        return "zh"
    if re.search(r"[A-Za-z]", text):
        return "en"
    return "zh"


def language_provider(provider, text, config):
    # Never change the shared provider: concurrent Chinese/English/Japanese
    # messages must keep their own voice. Non-Edge services retain their API.
    if not config.get("auto_language_voice", True) or not type(provider).__module__.endswith(
        "edge_tts_source"
    ):
        return provider
    language = {"中文": "zh", "英文": "en", "日语": "ja"}.get(
        config.get("speech_language"), speech_language(text)
    )
    chinese_voice = config.get("chinese_voice", "跟随提供商")
    if (
        language == "zh"
        and chinese_voice == "跟随提供商"
        and not config.get("chinese_rate")
        and not config.get("chinese_pitch")
    ):
        return provider
    selected = copy.copy(provider)
    if language == "zh":
        if chinese_voice in ("zh-CN-XiaoyiNeural", "zh-CN-XiaoxiaoNeural"):
            selected.voice = chinese_voice
        selected.rate = prosody(config.get("chinese_rate"), "%", provider.rate)
        selected.pitch = prosody(config.get("chinese_pitch"), "Hz", provider.pitch)
    elif language == "en":
        selected.voice = config.get("english_voice", "en-US-AnaNeural")
        selected.rate = prosody(config.get("english_rate"), "%", "+0%")
        selected.pitch = prosody(config.get("english_pitch"), "Hz", "+0Hz")
    else:
        selected.voice = config.get("japanese_voice", "ja-JP-NanamiNeural")
        selected.rate = prosody(config.get("japanese_rate"), "%", "+6%")
        selected.pitch = prosody(config.get("japanese_pitch"), "Hz", "+20Hz")
    return selected


def prosody(value, unit, fallback):
    # Invalid UI input must not turn an otherwise valid reply into a TTS error.
    value = str(value or "").strip()
    if re.fullmatch(r"[+-]\d+" + re.escape(unit), value):
        number = int(value[: -len(unit)])
        if (-50 if unit == "%" else -100) <= number <= 100:
            return value
    return fallback


def has_voice(event):
    if any(isinstance(p, Record) for p in event.get_messages()):
        return True
    # Native STT replaces Record with Plain before plugin dispatch. Inspect the
    # original top-level attachment metadata, excluding quoted messages.
    raw = getattr(getattr(event, "message_obj", None), "raw_message", None)
    if isinstance(raw, dict):
        raw = raw.get("d", raw)

    def field(obj, key, default=None):
        return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)

    segments = field(raw, "message", [])
    if isinstance(segments, list) and any(
        field(p, "type") in ("record", "audio") for p in segments
    ):
        return True
    attachments = field(raw, "attachments", []) or []
    for item in attachments if isinstance(attachments, list) else []:
        content_type = str(field(item, "content_type", "") or "").lower()
        name = str(field(item, "filename", "") or "").lower()
        if content_type.startswith(("audio", "voice")) or name.endswith(
            (".amr", ".silk", ".wav", ".mp3", ".ogg", ".m4a")
        ):
            return True
    return False


def reply_scene(event):
    remembered = event.get_extra("yueling_reply_scene")
    if remembered in ("mention", "automatic", "private"):
        return remembered
    if not event.get_group_id():
        return "private"
    bot_id = str(event.get_self_id() or "")
    # Wake words, @all, mentions of other members and quoted mentions are not
    # a direct mention of this bot. Capture before other plugins rewrite parts.
    if bot_id and any(isinstance(p, At) and str(p.qq) == bot_id for p in event.get_messages()):
        return "mention"
    return "automatic"


def speech_text(text, limit=600):
    text = re.sub(
        r"<(think|thinking|analysis|reasoning)\b[^>]*>.*?</\1\s*>", "", text, flags=re.I | re.S
    )
    text = re.sub(r"```.*?```", "，代码请看文字回复。", text, flags=re.S)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"https?://\S+", "链接请看文字回复", text)
    text = re.sub(r"(?m)^\s*(?:#{1,6}|[-*>]|\d+[.)、])\s*", "", text)
    text = re.sub(r"[*_`~#]+", "", text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = "\n".join(re.sub(r"[^\S\n]+", " ", line).strip() for line in text.splitlines()).strip()
    if len(text) > limit:
        text = text[:limit].rstrip("，、；： \n")
        language = speech_language(speech_segments(text)[-1])
        notice = {
            "ja": "続きはテキストをご覧ください。",
            "en": "Please see the text for the rest.",
            "zh": "其余内容请看文字。",
        }[language]
        text += "\n" + notice
    return text


def speech_segments(text):
    # Keep paragraph and sentence boundaries: the same answer may contain
    # Chinese, English and Japanese. Do not infer one voice for the whole reply.
    parts = [
        x.strip() for x in re.split(r"\n+|(?<=[。！？!?；;])\s*|(?<=[.])\s+", text) if x.strip()
    ]
    groups = []
    for part in parts:
        if groups and speech_language(groups[-1]) == speech_language(part):
            groups[-1] += " " + part
        else:
            groups.append(part)
    return groups


def concatenate_audio(paths):
    output = paths[0].parent / ("yueling-mixed-" + uuid.uuid4().hex + ".wav")
    try:
        with wave.open(str(output), "wb") as target:
            expected = None
            for path in paths:
                with wave.open(str(path), "rb") as source:
                    format_key = (
                        source.getnchannels(),
                        source.getsampwidth(),
                        source.getframerate(),
                        source.getcomptype(),
                    )
                    if expected is None:
                        target.setparams(source.getparams())
                        expected = format_key
                    if format_key != expected:
                        raise ValueError("合并语音的音频格式不一致")
                    while frames := source.readframes(16384):
                        target.writeframesraw(frames)
        return output
    except BaseException:
        output.unlink(missing_ok=True)
        raise


@register(
    "yueling_voice",
    "Kimari-0523",
    "自动回复与@回复独立设置文字、语音和随机；中英日分段配音",
    "1.3.0",
)
class YuelingVoice(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.gate = asyncio.Semaphore(2)
        self.cooldowns = {}
        logger.info("[柚铃语音] v1.3.0 已加载：自动回复设置、@回复设置独立，私聊沿用原有模式")

    def allowed(self, event):
        groups = [str(x).strip() for x in self.config.get("group_whitelist", []) if str(x).strip()]
        return self.config.get("enabled", True) and (
            not groups or str(event.get_group_id()) in groups
        )

    def reply_plan(self, event):
        scene = reply_scene(event)
        plan = {"voice": False, "voice_only": False, "scene": scene}
        if not self.allowed(event):
            return plan
        text = event.message_str or ""
        if NO_VOICE.search(text):
            return plan
        if self.config.get("separate_triggers", False) and scene != "private":
            settings = self.config.get(
                "mention_reply" if scene == "mention" else "automatic_reply", {}
            )
            mode = settings.get("mode", "仅文字" if scene == "mention" else "仅语音")
            wanted = mode in ("仅语音", "文字+语音")
            if mode == "随机":
                wanted = (
                    random.random()
                    < max(0, min(100, int(settings.get("random_probability", 20)))) / 100
                )
            plan.update(voice=wanted, voice_only=wanted and mode != "文字+语音")
            return plan
        if self.config.get("reply_mode", "按需") == "仅文字":
            return plan
        explicit = bool(WANT.search(text))
        incoming = has_voice(event) or bool(event.get_extra("yueling_incoming_voice"))
        mode = self.config.get("reply_mode", "按需")
        plan["voice"] = (
            explicit
            or incoming
            or mode == "每次"
            or (
                mode == "随机"
                and random.random()
                < max(0, min(100, int(self.config.get("random_probability", 20)))) / 100
            )
        )
        return plan

    def wants_voice(self, event):
        return self.reply_plan(event)["voice"]

    async def audio(self, event, text):
        provider = await self.context.get_using_tts_provider_async(event.unified_msg_origin)
        if provider is None:
            raise RuntimeError("未选择语音合成提供商")
        edge = type(provider).__module__.endswith("edge_tts_source")
        if edge:
            restore_edge_sdk_headers()
        parts = (
            speech_segments(text)
            if (
                edge
                and self.config.get("auto_language_voice", True)
                and self.config.get("speech_language", "自动识别") == "自动识别"
            )
            else [text]
        )

        async def synthesize(part):
            selected = language_provider(provider, part, self.config)
            async with self.gate:
                async with asyncio.timeout(35):
                    path = await selected.get_audio(part)
            if not path or not Path(path).is_file() or Path(path).stat().st_size == 0:
                raise RuntimeError("合成音频为空")
            return Path(path)

        async with asyncio.timeout(90):
            async with asyncio.TaskGroup() as group:
                jobs = [group.create_task(synthesize(part)) for part in parts]
        paths = [job.result() for job in jobs]
        path = concatenate_audio(paths) if len(paths) > 1 else paths[0]
        logger.info(
            "[柚铃语音] 合成完成，分段=%s，语言=%s，字数=%s",
            len(parts),
            ",".join(speech_language(part) for part in parts),
            len(text),
        )
        return Record(file=str(path), url=str(path), text=text)

    @filter.event_message_type(filter.EventMessageType.ALL, priority=900)
    async def remember_voice(self, event: AstrMessageEvent):
        event.set_extra("yueling_reply_scene", reply_scene(event))
        if has_voice(event):
            event.set_extra("yueling_incoming_voice", True)

    @filter.on_llm_request(priority=-1100)
    async def voice_request(self, event: AstrMessageEvent, request: ProviderRequest):
        # Preserve the existing prompt, history, images, tools and session identity.
        # The context rule also applies when just the voice-output switch is off.
        if self.config.get("context_guard_enabled", True) and CONTEXT_RULE not in (
            request.system_prompt or ""
        ):
            request.system_prompt = (request.system_prompt or "") + "\n" + CONTEXT_RULE
        # Decide once: random mode must not be sampled again when decorating.
        plan = self.reply_plan(event)
        event.set_extra("yueling_voice_plan", plan)
        event.set_extra("yueling_wants_voice", plan["voice"])
        if plan["voice"]:
            request.system_prompt = (
                request.system_prompt or ""
            ) + "\n这次回复会给本轮最终答复配语音，请自然回答当前问题，不写动作描写或内部思考。"

    @filter.on_decorating_result(priority=-2000)
    async def voice_reply(self, event: AstrMessageEvent):
        result = event.get_result()
        if (
            not result
            or not result.is_llm_result()
            or event.get_extra("yueling_voice_done")
            or not self.allowed(event)
        ):
            return
        if any(isinstance(p, Record) for p in result.chain):
            return
        plan = event.get_extra("yueling_voice_plan")
        if plan is None:
            plan = self.reply_plan(event)
            event.set_extra("yueling_voice_plan", plan)
        if not plan["voice"] or not await SessionServiceManager.should_process_tts_request(event):
            return
        key = (event.unified_msg_origin, plan["scene"])
        now = time.monotonic()
        if now - self.cooldowns.get(key, -30) < 5:
            return
        limit = max(80, min(1200, int(self.config.get("max_speech_chars", 600))))
        original = "".join(p.text for p in result.chain if isinstance(p, Plain))
        text = speech_text(original, limit)
        if not text:
            return
        event.set_extra("yueling_voice_done", True)
        self.cooldowns[key] = now
        if len(self.cooldowns) > 1024:
            self.cooldowns = {k: v for k, v in self.cooldowns.items() if now - v < 60}
        try:
            part = await self.audio(event, text)
            # Only remove text after successful synthesis. Long answers retain
            # full text when speech was truncated; other components stay intact.
            if plan["voice_only"] and len(speech_text(original, 1000000)) <= limit:
                result.chain[:] = [p for p in result.chain if not isinstance(p, Plain)]
            result.chain.append(part)
            result.use_t2i(False)
            logger.info(
                "[柚铃语音] 已生成回复语音，场景=%s，仅语音=%s，字数=%s",
                plan["scene"],
                plan["voice_only"],
                len(text),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("[柚铃语音] 合成失败，保留文字回复：%s", type(exc).__name__)

    @filter.command("语音")
    async def say(self, event: AstrMessageEvent, text: str = ""):
        if not self.allowed(event):
            return
        if (
            not self.config.get("separate_triggers", False)
            and self.config.get("reply_mode", "按需") == "仅文字"
        ):
            yield event.plain_result("当前为仅文字模式，可在“柚铃语音助手”配置中切换语音回复方式。")
            return
        # Command parser's string argument may contain just its first token.
        original = (event.message_str or "").strip()
        match = re.search(r"(?:^|\s)/?语音(?:\s+|$)(.*)", original, re.S)
        text = match.group(1).strip() if match else text.strip()
        if not text:
            yield event.plain_result("用法：/语音 你好呀，我是柚铃。也可以@我说“用语音回复”。")
            return
        text = speech_text(text, max(80, min(1200, int(self.config.get("max_speech_chars", 600)))))
        if not text:
            yield event.plain_result("请提供需要朗读的文字。")
            return
        try:
            part = await self.audio(event, text)
            yield event.chain_result([part]).use_t2i(False)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("[柚铃语音] 指令合成失败：%s", type(exc).__name__)
            yield event.plain_result("这次语音合成失败了，请稍后再试。")
