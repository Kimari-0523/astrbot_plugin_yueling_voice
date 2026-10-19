import asyncio
import contextlib
import importlib
import importlib.util
import sys
import types
import unittest
import uuid
import wave
from pathlib import Path
from unittest.mock import AsyncMock, patch

if importlib.util.find_spec("astrbot") is None:

    def module(name, **values):
        obj = types.ModuleType(name)
        obj.__dict__.update(values)
        sys.modules[name] = obj
        return obj

    def decorator(*args, **kwargs):
        return lambda func: func

    class Component:
        def __init__(self, text="", **kwargs):
            self.text = text
            self.__dict__.update(kwargs)

    class Plain(Component):
        pass

    class Record(Component):
        pass

    class At(Component):
        pass

    class Star:
        def __init__(self, context):
            self.context = context

    module("astrbot")
    module(
        "astrbot.api",
        AstrBotConfig=dict,
        logger=types.SimpleNamespace(info=lambda *a: None, warning=lambda *a: None),
    )
    module(
        "astrbot.api.event",
        AstrMessageEvent=object,
        filter=types.SimpleNamespace(
            event_message_type=decorator,
            on_llm_request=decorator,
            on_decorating_result=decorator,
            command=decorator,
            EventMessageType=types.SimpleNamespace(ALL=1),
        ),
    )
    module("astrbot.api.message_components", Plain=Plain, Record=Record, At=At)
    module("astrbot.api.provider", ProviderRequest=object)
    module("astrbot.api.star", Context=object, Star=Star, register=decorator)
    module("astrbot.core")
    module("astrbot.core.star")
    module(
        "astrbot.core.star.session_llm_manager",
        SessionServiceManager=types.SimpleNamespace(
            should_process_tts_request=AsyncMock(return_value=True)
        ),
    )

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
voice = importlib.import_module(Path(__file__).parent.name + ".main")


class Result:
    def __init__(self):
        self.chain = [voice.Plain(text="这是正常的最终回答。")]
        self.t2i = True

    def is_llm_result(self):
        return True

    def use_t2i(self, value):
        self.t2i = value
        return self


class Event:
    def __init__(self, text="", parts=None):
        self.message_str = text
        self.parts = parts or []
        self.extra = {}
        self.unified_msg_origin = "platform:group:123"
        self.result = Result()

    def get_group_id(self):
        return "123"

    def get_self_id(self):
        return "bot"

    def get_messages(self):
        return self.parts

    def get_extra(self, key):
        return self.extra.get(key)

    def set_extra(self, key, value):
        self.extra[key] = value

    def get_result(self):
        return self.result


class VoiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.plugin = voice.YuelingVoice(types.SimpleNamespace(), {})
        self.plugin.audio = AsyncMock(return_value=voice.Record(file="/tmp/test.wav"))
        voice.SessionServiceManager.should_process_tts_request = AsyncMock(return_value=True)

    async def test_plain_chat_stays_text(self):
        event = Event("你好")
        await self.plugin.voice_reply(event)
        self.plugin.audio.assert_not_awaited()
        self.assertEqual(len(event.result.chain), 1)

    async def test_voice_input_and_explicit_request_attach_audio_without_removing_text(self):
        for event in [
            Event(parts=[voice.Record(file="/tmp/input.wav")]),
            Event("用语音回复，今天开心吗？"),
        ]:
            self.plugin.cooldowns.clear()
            await self.plugin.voice_reply(event)
            self.assertEqual(len(event.result.chain), 2)
            self.assertIsInstance(event.result.chain[0], voice.Plain)
            self.assertIsInstance(event.result.chain[1], voice.Record)
            self.assertFalse(event.result.t2i)
            await self.plugin.voice_reply(event)
            self.assertEqual(len(event.result.chain), 2)

    async def test_failure_keeps_original_reply(self):
        event = Event("请用语音回复")
        self.plugin.audio.side_effect = RuntimeError("failed")
        await self.plugin.voice_reply(event)
        self.assertEqual(event.result.chain[0].text, "这是正常的最终回答。")
        self.assertEqual(len(event.result.chain), 1)

    async def test_opt_out_whitelist_and_session_disable(self):
        self.assertFalse(
            self.plugin.wants_voice(Event("不要用语音回复", [voice.Record(file="/tmp/input.wav")]))
        )
        self.plugin.config = {"group_whitelist": ["other"]}
        self.assertFalse(self.plugin.wants_voice(Event("用语音回答")))
        self.plugin.config = {}
        voice.SessionServiceManager.should_process_tts_request.return_value = False
        await self.plugin.voice_reply(Event("用语音回答"))
        self.plugin.audio.assert_not_awaited()

    async def test_request_preserves_prompt_history_and_current_question(self):
        history = [
            {"role": "user", "content": "接下来只聊初音未来。"},
            {"role": "assistant", "content": "好的，我们聊初音未来。"},
        ]
        event = Event("用语音回复，它的英文名呢？")
        request = types.SimpleNamespace(
            system_prompt="原人格",
            contexts=history,
            prompt="它的英文名呢？",
            tools=[{"name": "existing_tool"}],
        )
        await self.plugin.voice_request(event, request)
        self.assertTrue(event.extra["yueling_wants_voice"])
        self.assertIn("原人格", request.system_prompt)
        self.assertIn(voice.CONTEXT_RULE, request.system_prompt)
        self.assertIs(request.contexts, history)
        self.assertEqual(request.prompt, "它的英文名呢？")
        self.assertEqual(request.tools, [{"name": "existing_tool"}])

    async def test_turning_off_voice_keeps_context_rule_and_normal_text(self):
        self.plugin.config = {"enabled": False}
        event = Event("用语音回复")
        request = types.SimpleNamespace(system_prompt="原人格")
        await self.plugin.voice_request(event, request)
        await self.plugin.voice_reply(event)
        self.plugin.audio.assert_not_awaited()
        self.assertIn(voice.CONTEXT_RULE, request.system_prompt)
        self.assertEqual(len(event.result.chain), 1)
        self.plugin.config = {"reply_mode": "仅文字"}
        self.assertFalse(
            self.plugin.wants_voice(Event("用语音回复", [voice.Record(file="/tmp/in.wav")]))
        )

    async def test_context_rule_can_be_disabled_and_is_not_duplicated(self):
        event = Event("你好")
        request = types.SimpleNamespace(system_prompt="原人格")
        await self.plugin.voice_request(event, request)
        await self.plugin.voice_request(event, request)
        self.assertEqual(request.system_prompt.count(voice.CONTEXT_RULE), 1)
        self.plugin.config = {"context_guard_enabled": False}
        request.system_prompt = "原人格"
        await self.plugin.voice_request(event, request)
        self.assertEqual(request.system_prompt, "原人格")

    async def test_concurrent_replies_use_each_events_own_final_answer(self):
        first = Event("用语音回复")
        second = Event("用语音回复")
        first.unified_msg_origin = "platform:group:first"
        second.unified_msg_origin = "platform:group:second"
        first.result.chain = [voice.Plain(text="第一群刚才说的是初音未来。")]
        second.result.chain = [voice.Plain(text="第二群当前问的是语音开关。")]

        async def synthesize(event, text):
            await asyncio.sleep(0)
            return voice.Record(file="/tmp/test.wav", text=text)

        self.plugin.audio = AsyncMock(side_effect=synthesize)
        await asyncio.gather(self.plugin.voice_reply(first), self.plugin.voice_reply(second))
        self.assertEqual(first.result.chain[-1].text, "第一群刚才说的是初音未来。")
        self.assertEqual(second.result.chain[-1].text, "第二群当前问的是语音开关。")

    async def test_different_modes_do_not_remove_explicit_opt_out(self):
        self.plugin.config = {"reply_mode": "每次"}
        self.assertTrue(self.plugin.wants_voice(Event("你好")))
        self.assertFalse(self.plugin.wants_voice(Event("只用文字回答")))
        self.plugin.config = {"reply_mode": "随机", "random_probability": 0}
        self.assertFalse(self.plugin.wants_voice(Event("你好")))
        self.assertTrue(self.plugin.wants_voice(Event("用语音回答")))

    async def test_native_stt_replaced_record_still_detects_original_voice(self):
        for raw in [
            {"attachments": [{"content_type": "audio/amr", "filename": "message.amr"}]},
            types.SimpleNamespace(
                attachments=[types.SimpleNamespace(content_type="voice", filename="voice.amr")]
            ),
            {"message": [{"type": "record", "data": {"file": "voice.silk"}}]},
        ]:
            event = Event("这是识别后的文字", [voice.Plain(text="这是识别后的文字")])
            event.message_obj = types.SimpleNamespace(raw_message=raw)
            self.assertTrue(self.plugin.wants_voice(event))
        event.message_obj.raw_message = {
            "attachments": [{"content_type": "image/png", "filename": "photo.png"}]
        }
        self.assertFalse(self.plugin.wants_voice(event))

    async def test_markup_links_code_and_long_text_are_not_read_verbatim(self):
        cleaned = voice.speech_text(
            "<think>不该读的思考</think>**你好** [文档](https://example.com) ```python\nsecret=1\n```"
        )
        self.assertNotIn("思考", cleaned)
        self.assertNotIn("secret", cleaned)
        self.assertNotIn("https", cleaned)
        self.assertIn("你好", cleaned)
        self.assertLess(len(voice.speech_text("你好" * 200, 160)), 180)

    async def test_edge_compatibility_uses_sdk_version_not_hardcoded_browser(self):
        constants = types.ModuleType("edge_tts.constants")
        constants.BASE_HEADERS = {"User-Agent": "SDK-current-version"}
        constants.WSS_HEADERS = {"User-Agent": "astrbot/old", "Origin": "sdk-origin"}
        with patch.dict(
            sys.modules, {"edge_tts": types.ModuleType("edge_tts"), "edge_tts.constants": constants}
        ):
            voice.restore_edge_sdk_headers()
        self.assertEqual(
            constants.WSS_HEADERS, {"User-Agent": "SDK-current-version", "Origin": "sdk-origin"}
        )

    async def test_language_detection_uses_answer_and_does_not_treat_brands_as_english(self):
        for text, expected in [
            ("你好呀，我是柚铃。", "zh"),
            ("Hello, I am Yueling!", "en"),
            ("こんにちは、ゆうりんです。今日もよろしくね。", "ja"),
            ("GPT 和 Claude 都能回答中文。", "zh"),
            ("初音未来", "zh"),
            ("123!?", "zh"),
        ]:
            self.assertEqual(voice.speech_language(text), expected)

    async def test_language_selection_keeps_chinese_and_shared_provider_unchanged(self):
        provider = type("Edge", (), {"__module__": "test.edge_tts_source"})()
        provider.voice = "zh-CN-XiaoyiNeural"
        provider.rate = "+4%"
        provider.pitch = "+12Hz"
        provider.volume = "+0%"
        self.assertIs(voice.language_provider(provider, "你好", {}), provider)
        english = voice.language_provider(provider, "Hello, how are you?", {})
        japanese = voice.language_provider(provider, "こんにちは、今日はいい天気ですね。", {})
        self.assertEqual(english.voice, "en-US-AnaNeural")
        self.assertEqual(english.pitch, "+0Hz")
        self.assertEqual(japanese.voice, "ja-JP-NanamiNeural")
        self.assertEqual(provider.voice, "zh-CN-XiaoyiNeural")
        self.assertEqual(provider.pitch, "+12Hz")
        self.assertIsNot(english, japanese)

    async def test_japanese_choice_disable_and_non_edge_provider(self):
        provider = type("Edge", (), {"__module__": "test.edge_tts_source"})()
        selected = voice.language_provider(
            provider,
            "こんにちは",
            {
                "japanese_voice": "en-US-EmmaMultilingualNeural",
                "japanese_pitch": "+0Hz",
                "japanese_rate": "-5%",
            },
        )
        self.assertEqual(
            (selected.voice, selected.pitch, selected.rate),
            ("en-US-EmmaMultilingualNeural", "+0Hz", "-5%"),
        )
        self.assertIs(
            voice.language_provider(provider, "Hello", {"auto_language_voice": False}), provider
        )
        other = types.SimpleNamespace(voice="other")
        self.assertIs(voice.language_provider(other, "Hello", {}), other)

    async def test_manual_language_and_each_language_controls_do_not_mutate_provider(self):
        provider = type("Edge", (), {"__module__": "test.edge_tts_source"})()
        provider.voice = "zh-CN-XiaoyiNeural"
        provider.rate = "+4%"
        provider.pitch = "+12Hz"
        for config, text, expected in [
            (
                {
                    "chinese_voice": "zh-CN-XiaoxiaoNeural",
                    "chinese_rate": "-5%",
                    "chinese_pitch": "+3Hz",
                },
                "你好",
                ("zh-CN-XiaoxiaoNeural", "-5%", "+3Hz"),
            ),
            (
                {
                    "english_voice": "en-US-JennyNeural",
                    "english_rate": "+5%",
                    "english_pitch": "+10Hz",
                },
                "Hello",
                ("en-US-JennyNeural", "+5%", "+10Hz"),
            ),
            ({"speech_language": "日语"}, "初音未来", ("ja-JP-NanamiNeural", "+6%", "+20Hz")),
            ({"speech_language": "英文"}, "你好", ("en-US-AnaNeural", "+0%", "+0Hz")),
        ]:
            selected = voice.language_provider(provider, text, config)
            self.assertEqual((selected.voice, selected.rate, selected.pitch), expected)
            self.assertEqual(
                (provider.voice, provider.rate, provider.pitch),
                ("zh-CN-XiaoyiNeural", "+4%", "+12Hz"),
            )

    async def test_invalid_prosody_falls_back_instead_of_breaking_tts(self):
        for value in ["", None, "cute", "20Hz", "+999Hz", "+20%", "<tag>"]:
            self.assertEqual(voice.prosody(value, "Hz", "+12Hz"), "+12Hz")
        self.assertEqual(voice.prosody("-5%", "%", "+0%"), "-5%")
        self.assertEqual(voice.prosody("+999%", "%", "+0%"), "+0%")

    async def test_separate_triggers_auto_voice_only_and_bot_mention_text_only(self):
        self.plugin.config = {"separate_triggers": True}
        automatic = Event("今天聊点什么？")
        await self.plugin.voice_reply(automatic)
        self.assertEqual(len(automatic.result.chain), 1)
        self.assertIsInstance(automatic.result.chain[0], voice.Record)
        for event in [
            Event("你好", [voice.At(qq="bot")]),
            Event("用语音回答", [voice.At(qq="bot")]),
        ]:
            self.plugin.audio.reset_mock()
            await self.plugin.voice_reply(event)
            self.plugin.audio.assert_not_awaited()
            self.assertIsInstance(event.result.chain[0], voice.Plain)

    async def test_direct_mention_excludes_other_members_all_text_and_nested_quotes(self):
        self.plugin.config = {"separate_triggers": True}
        for parts in [
            [voice.At(qq="other")],
            [voice.At(qq="all")],
            [voice.Plain(text="@bot")],
            [types.SimpleNamespace(chain=[voice.At(qq="bot")])],
        ]:
            self.assertEqual(voice.reply_scene(Event(parts=parts)), "automatic")
        event = Event(parts=[voice.At(qq="bot")])
        await self.plugin.remember_voice(event)
        event.parts = []
        self.assertEqual(voice.reply_scene(event), "mention")

    async def test_independent_random_probabilities_and_one_sample_per_answer(self):
        self.plugin.config = {
            "separate_triggers": True,
            "automatic_reply": {"mode": "随机", "random_probability": 80},
            "mention_reply": {"mode": "随机", "random_probability": 20},
        }
        with patch.object(voice.random, "random", return_value=0.5):
            self.assertTrue(self.plugin.wants_voice(Event("你好")))
            self.assertFalse(self.plugin.wants_voice(Event("你好", [voice.At(qq="bot")])))
        event = Event("你好", [voice.At(qq="bot")])
        with patch.object(voice.random, "random", side_effect=[0.1, 0.9]) as sample:
            await self.plugin.voice_request(event, types.SimpleNamespace(system_prompt="人格"))
            await self.plugin.voice_reply(event)
            self.assertEqual(sample.call_count, 1)
        self.assertEqual(len(event.result.chain), 1)
        self.assertIsInstance(event.result.chain[0], voice.Record)

    async def test_voice_only_failure_opt_out_and_private_chat_keep_text_behavior(self):
        self.plugin.config = {"separate_triggers": True}
        event = Event("你好")
        self.plugin.audio.side_effect = RuntimeError("offline")
        await self.plugin.voice_reply(event)
        self.assertIsInstance(event.result.chain[0], voice.Plain)
        self.assertFalse(self.plugin.wants_voice(Event("只用文字回答")))
        private = Event("你好")
        private.get_group_id = lambda: ""
        self.assertEqual(voice.reply_scene(private), "private")
        self.assertFalse(self.plugin.wants_voice(private))
        private.message_str = "用语音回答"
        self.assertTrue(self.plugin.wants_voice(private))

    async def test_separate_text_plus_voice_and_truncated_voice_keep_full_text(self):
        self.plugin.config = {"separate_triggers": True, "automatic_reply": {"mode": "文字+语音"}}
        event = Event("你好")
        await self.plugin.voice_reply(event)
        self.assertEqual(len(event.result.chain), 2)
        self.assertIsInstance(event.result.chain[0], voice.Plain)
        self.plugin.cooldowns.clear()
        self.plugin.config = {"separate_triggers": True, "max_speech_chars": 80}
        event = Event("你好")
        event.result.chain = [voice.Plain(text="这是一段很长的回答。" * 30)]
        await self.plugin.voice_reply(event)
        self.assertEqual(len(event.result.chain), 2)
        self.assertEqual(event.result.chain[0].text, "这是一段很长的回答。" * 30)

    async def test_mixed_regression_keeps_japanese_after_long_english_paragraph(self):
        raw = (
            "银行行长说，这行文字的行，读音很容易搞错；石榴、樱桃、初音未来，都是日常好听的词。\n\u200b\n"
            "The irregular rhythm, unusual vocabulary and loanwords are tricky for speech synthesis engines.\n\u200b\n"
            "今日は「はし」（橋・箸）のアクセントや、外来語「コンピューター」、難読漢字「叡智」をテストします。"
        )
        cleaned = voice.speech_text(raw)
        self.assertGreater(len(cleaned), 160)
        self.assertIn("テストします。", cleaned)
        self.assertNotIn("其余内容", cleaned)
        self.assertEqual(
            [voice.speech_language(part) for part in voice.speech_segments(cleaned)],
            ["zh", "en", "ja"],
        )

    async def test_japanese_truncation_uses_japanese_notice(self):
        text = voice.speech_text("こんにちは、今日はいい天気ですね。" * 40, 160)
        self.assertNotIn("其余内容", text)
        self.assertTrue(text.endswith("続きはテキストをご覧ください。"))

    async def test_mixed_audio_uses_three_voices_and_preserves_frame_order(self):
        calls = []
        with contextlib.ExitStack() as cleanup:
            directory = Path(__file__).resolve().parent
            prefix = uuid.uuid4().hex

            async def get_audio(provider, text):
                calls.append((provider.voice, text))
                number = {"zh-CN-XiaoyiNeural": 1, "en-US-AnaNeural": 2, "ja-JP-NanamiNeural": 3}[
                    provider.voice
                ]
                path = directory / (prefix + "-" + str(number) + ".wav")
                cleanup.callback(path.unlink, missing_ok=True)
                with wave.open(str(path), "wb") as output:
                    output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                    output.writeframes(number.to_bytes(2, "little") * 100)
                await asyncio.sleep(0)
                return str(path)

            provider = type(
                "Edge", (), {"__module__": "test.edge_tts_source", "get_audio": get_audio}
            )()
            provider.voice = "zh-CN-XiaoyiNeural"
            provider.rate = "+4%"
            provider.pitch = "+12Hz"
            context = types.SimpleNamespace(
                get_using_tts_provider_async=AsyncMock(return_value=provider)
            )
            plugin = voice.YuelingVoice(context, {})
            with patch.object(voice, "restore_edge_sdk_headers"):
                record = await plugin.audio(
                    Event(), "你好。\nHello, nice to meet you!\nこんにちは、よろしくね。"
                )
            cleanup.callback(Path(record.file).unlink, missing_ok=True)
            self.assertEqual(
                [name for name, _ in calls],
                ["zh-CN-XiaoyiNeural", "en-US-AnaNeural", "ja-JP-NanamiNeural"],
            )
            with wave.open(record.file, "rb") as output:
                self.assertEqual(output.getnframes(), 300)
                self.assertEqual(
                    output.readframes(300),
                    b"\x01\x00" * 100 + b"\x02\x00" * 100 + b"\x03\x00" * 100,
                )
            self.assertEqual(provider.voice, "zh-CN-XiaoyiNeural")


if __name__ == "__main__":
    unittest.main()
