import asyncio
import logging
from typing import Optional, Callable
import numpy as np
from scipy.signal import resample_poly
from transcribe import TranscriptionProcessor

logger = logging.getLogger(__name__)


class AudioInputProcessor:
    """
    Manages audio input, processes it for transcription, and handles related callbacks.

    This class receives raw audio chunks, resamples them to the required format (16kHz),
    feeds them to an underlying `TranscriptionProcessor`, and manages callbacks for
    real-time transcription updates, recording start events, and silence detection.
    It also runs the transcription process in a background task.
    """

    # _RESAMPLE_RATIO = 3  # This was incorrect for 24kHz -> 16kHz.
    # Client sends 24kHz. Target is 16kHz.
    # Resampling factor: 16000 / 24000 = 2 / 3
    _RESAMPLE_UP = 2
    _RESAMPLE_DOWN = 3

    def __init__(
            self,
            language: str = "en",
            is_orpheus: bool = False,
            silence_active_callback: Optional[Callable[[bool], None]] = None,
            speech_start_server_callback: Optional[Callable[[], None]] = None, # New callback
            pipeline_latency: float = 0.5,
        ) -> None:
        """
        Initializes the AudioInputProcessor.

        Args:
            language: Target language code for transcription (e.g., "en").
            is_orpheus: Flag indicating if a specific model variant should be used.
            silence_active_callback: Optional callback function invoked when silence state changes.
                                     It receives a boolean argument (True if silence is active).
            speech_start_server_callback: Optional callback for when speech starts, to signal server. # New callback doc
            pipeline_latency: Estimated latency of the processing pipeline in seconds.
        """
        self.last_partial_text: Optional[str] = None
        self.active_speech_start_notifier: Optional[Callable[[], None]] = None
        self.active_realtime_callback: Optional[Callable[[str], None]] = None
        self.active_potential_sentence_callback: Optional[Callable[[str], None]] = None
        self.active_tts_allowed_callback: Optional[Callable[[], None]] = None
        self.active_potential_final_callback: Optional[Callable[[str], None]] = None
        self.active_potential_abort_callback: Optional[Callable[[], None]] = None
        self.active_final_transcription_callback: Optional[Callable[[str], None]] = None
        self.active_before_final_sentence_callback: Optional[Callable[[Optional[np.ndarray], Optional[str]], bool]] = None
        self.active_recording_start_callback: Optional[Callable[[], None]] = None
        self.active_silence_active_callback: Optional[Callable[[bool], None]] = None

        self.transcriber = TranscriptionProcessor(
            source_language=language, # Corrected keyword argument
            # Callbacks for TranscriptionProcessor are internal methods of AudioInputProcessor
            realtime_transcription_callback=self._internal_on_partial,
            full_transcription_callback=self._internal_on_final,
            potential_full_transcription_callback=self._internal_on_potential_final,
            potential_full_transcription_abort_callback=self._internal_on_potential_abort,
            potential_sentence_end=self._internal_on_potential_sentence,
            before_final_sentence=self._internal_on_before_final,
            silence_active_callback=self._internal_on_silence_active,
            on_recording_start_callback=self._internal_on_recording_start,
            on_speech_start_utterance_callback=self._internal_on_speech_start_utterance,
            on_tts_allowed_to_synthesize=self._internal_on_tts_allowed, # Pass new internal handler
            is_orpheus=is_orpheus,
            pipeline_latency=pipeline_latency,
        )
        # Flag to indicate if the transcription loop has failed fatally
        self._transcription_failed = False
        # self.transcription_task = asyncio.create_task(self._run_transcription_loop()) # Removed

        self.interrupted = False
        logger.info("👂🚀 AudioInputProcessor initialized.")

    def set_active_listeners(
        self,
        speech_start_notifier: Optional[Callable[[], None]] = None,
        realtime_callback: Optional[Callable[[str], None]] = None,
        potential_sentence_callback: Optional[Callable[[str], None]] = None,
        tts_allowed_callback: Optional[Callable[[], None]] = None,
        potential_final_callback: Optional[Callable[[str], None]] = None,
        potential_abort_callback: Optional[Callable[[], None]] = None,
        final_transcription_callback: Optional[Callable[[str], None]] = None,
        before_final_sentence_callback: Optional[Callable[[Optional[np.ndarray], Optional[str]], bool]] = None,
        recording_start_callback: Optional[Callable[[], None]] = None,
        silence_active_callback: Optional[Callable[[bool], None]] = None
    ):
        """Sets or clears all active listener callbacks for the current WebSocket connection."""
        self.active_speech_start_notifier = speech_start_notifier
        self.active_realtime_callback = realtime_callback
        self.active_potential_sentence_callback = potential_sentence_callback
        self.active_tts_allowed_callback = tts_allowed_callback
        self.active_potential_final_callback = potential_final_callback
        self.active_potential_abort_callback = potential_abort_callback
        self.active_final_transcription_callback = final_transcription_callback
        self.active_before_final_sentence_callback = before_final_sentence_callback
        self.active_recording_start_callback = recording_start_callback
        self.active_silence_active_callback = silence_active_callback
        logger.info("👂🔔 Active listeners updated for AudioInputProcessor.")

    def clear_active_listeners(self):
        """Clears all active listener callbacks."""
        self.set_active_listeners() # Call with all None
        logger.info("👂🔕 Active listeners cleared for AudioInputProcessor.")

    # --- Internal relay methods to call active callbacks ---
    def _internal_on_partial(self, text: str) -> None:
        if text != self.last_partial_text:
            self.last_partial_text = text
            if self.active_realtime_callback:
                self.active_realtime_callback(text)

    def _internal_on_final(self, text: str) -> None:
        if self.active_final_transcription_callback:
            self.active_final_transcription_callback(text)

    def _internal_on_potential_final(self, text: str) -> None:
        if self.active_potential_final_callback:
            self.active_potential_final_callback(text)

    def _internal_on_potential_abort(self) -> None:
        if self.active_potential_abort_callback:
            self.active_potential_abort_callback()

    def _internal_on_potential_sentence(self, text: str) -> None:
        if self.active_potential_sentence_callback:
            self.active_potential_sentence_callback(text)

    def _internal_on_before_final(self, audio: Optional[np.ndarray], text: Optional[str]) -> bool:
        if self.active_before_final_sentence_callback:
            return self.active_before_final_sentence_callback(audio, text)
        return False

    def _internal_on_silence_active(self, is_active: bool) -> None:
        if self.active_silence_active_callback:
            self.active_silence_active_callback(is_active)

    def _internal_on_recording_start(self) -> None:
        if self.active_recording_start_callback:
            self.active_recording_start_callback()

    def _internal_on_tts_allowed(self) -> None:
        if self.active_tts_allowed_callback:
            self.active_tts_allowed_callback()

    def _internal_on_speech_start_utterance(self) -> None:
        if self.active_speech_start_notifier:
            logger.info("👂🚀 Relaying speech_start notification via active_speech_start_notifier.")
            self.active_speech_start_notifier()
        else:
            logger.debug("👂🔇 Speech started (from transcriber) but no active_speech_start_notifier set.")

    # --- Public methods ---
    def abort_generation(self) -> None:
        """Signals the underlying transcriber to abort any ongoing generation process."""
        logger.info("👂🛑 Aborting generation requested.")
        self.transcriber.abort_generation() # This now also re-creates the recorder and its loop

    # _run_transcription_loop method removed

    def process_audio_chunk(self, raw_bytes: bytes) -> np.ndarray:
        """
        Converts raw audio bytes (int16) to a 16kHz 16-bit PCM numpy array.

        The audio is converted to float32 for accurate resampling and then
        converted back to int16, clipping values outside the valid range.

        Args:
            raw_bytes: Raw audio data assumed to be in int16 format.

        Returns:
            A numpy array containing the resampled audio in int16 format at 16kHz.
            Returns an array of zeros if the input is silent.
        """
        raw_audio = np.frombuffer(raw_bytes, dtype=np.int16)

        if np.max(np.abs(raw_audio)) == 0:
            # Calculate expected length after resampling for silence
            expected_len = int(np.ceil(len(raw_audio) / self._RESAMPLE_RATIO))
            return np.zeros(expected_len, dtype=np.int16)

        # Convert to float32 for resampling precision
        audio_float32 = raw_audio.astype(np.float32)

        # Resample using float32 data
        resampled_float = resample_poly(audio_float32, self._RESAMPLE_UP, self._RESAMPLE_DOWN)

        # Convert back to int16, clipping to ensure validity
        resampled_int16 = np.clip(resampled_float, -32768, 32767).astype(np.int16)

        input_sample_rate = 24000 # Client sample rate
        output_sample_rate = input_sample_rate * self._RESAMPLE_UP / self._RESAMPLE_DOWN
        logger.debug(f"🎧 Resampled audio chunk from {input_sample_rate}Hz to {output_sample_rate:.0f} Hz, {len(raw_audio)} samples to {len(resampled_int16)} samples.")
        return resampled_int16


    async def process_chunk_queue(self, audio_queue: asyncio.Queue) -> None:
        """
        Continuously processes audio chunks received from an asyncio Queue.

        Retrieves audio data, processes it using `process_audio_chunk`, and
        feeds the result to the transcriber unless interrupted or the transcription
        task has failed. Stops when `None` is received from the queue or upon error.

        Args:
            audio_queue: An asyncio queue expected to yield dictionaries containing
                         'pcm' (raw audio bytes) or None to terminate.
        """
        logger.info("👂▶️ Starting audio chunk processing loop.")
        chunk_counter = 0
        last_log_time = time.time()
        
        while True:
            try:
                # 每10秒记录一次状态
                current_time = time.time()
                if current_time - last_log_time > 10:
                    logger.info(f"👂🔄 音频处理循环仍在运行，已处理 {chunk_counter} 个音频块")
                    last_log_time = current_time
                
                # Check if the transcription task (now managed by TranscriptionProcessor) has permanently failed.
                # This check might need adjustment as _transcription_failed is local to AudioInputProcessor
                # and not directly aware of the state of TranscriptionProcessor's new transcription_future.
                # For now, we rely on feed_audio being robust if the recorder inside transcriber is None temporarily.
                # A more robust solution might involve the transcriber exposing a health status.
                if not self.transcriber or self.transcriber.shutdown_performed: # Check if transcriber itself is shut down
                    logger.warning("👂⏹️ Transcriber is shut down. Stopping audio processing.")
                    break
                if self._transcription_failed: # This flag is still relevant if _run_transcription_loop (now removed) failed
                    logger.error("👂🛑 AudioInputProcessor's transcription management indicated a past failure. Stopping audio processing.")
                    break


                logger.debug("👂⏳ 等待音频队列中的数据...")
                audio_data = await audio_queue.get()
                if audio_data is None:
                    logger.info("👂🔌 Received termination signal for audio processing.")
                    break  # Termination signal

                chunk_counter += 1
                logger.debug(f"👂📦 收到第 {chunk_counter} 个音频块")
                
                pcm_data = audio_data.pop("pcm")
                logger.debug(f"👂📊 音频块大小: {len(pcm_data)} 字节")

                # Process audio chunk (resampling happens consistently via float32)
                processed = self.process_audio_chunk(pcm_data)
                if processed.size == 0:
                    logger.debug("👂🔇 跳过空音频块")
                    continue # Skip empty chunks

                # Feed audio only if not interrupted and transcriber should be running
                if not self.interrupted:
                    logger.debug(f"👂🎤 处理音频块 {chunk_counter}，大小: {processed.size} 样本")
                    # Check failure flag again, as it might have been set between queue.get and here
                    if not self._transcription_failed:
                        # Feed audio to the underlying processor
                        logger.debug(f"👂➡️ 将音频块 {chunk_counter} 传递给转录器")
                        self.transcriber.feed_audio(processed.tobytes(), audio_data)
                    else:
                        logger.warning(f"👂⚠️ 转录任务已失败，跳过音频块 {chunk_counter}")
                else:
                    logger.debug(f"👂⏸️ 音频处理被中断，跳过音频块 {chunk_counter}")

            except asyncio.CancelledError:
                logger.info("👂🚫 Audio processing task cancelled.")
                break
            except Exception as e:
                # Log general errors during audio chunk processing
                logger.error(f"👂💥 Audio processing error in queue loop: {e}", exc_info=True)
                # Continue processing subsequent chunks after logging the error.
                # Consider adding logic to break if errors persist.
        logger.info("👂⏹️ Audio chunk processing loop finished.")


    def shutdown(self) -> None:
        """
        Initiates shutdown procedures for the audio processor and transcriber.

        Signals the transcriber to shut down and cancels the background
        transcription task.
        """
        logger.info("👂🛑 Shutting down AudioInputProcessor...")
        # Ensure transcriber shutdown is called first to signal the loop
        if hasattr(self.transcriber, 'shutdown'):
             logger.info("👂🛑 Signaling TranscriptionProcessor to shut down.")
             self.transcriber.shutdown() # TranscriptionProcessor's shutdown now handles its future
        else:
             logger.warning("👂⚠️ TranscriptionProcessor does not have a shutdown method.")

        # self.transcription_task is removed, its cancellation is now handled by self.transcriber.shutdown()
        logger.info("👂👋 AudioInputProcessor shutdown sequence initiated.")