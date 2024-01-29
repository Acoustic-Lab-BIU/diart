from __future__ import annotations

from typing import Sequence, Generator

import numpy as np
import torch
from pyannote.core import Annotation, SlidingWindowFeature, SlidingWindow, Segment, Timeline
from pyannote.metrics.base import BaseMetric
from pyannote.metrics.diarization import DiarizationErrorRate
from typing_extensions import Literal

from . import base
from .aggregation import DelayedAggregation
from .clustering import OnlineSpeakerClustering
from .embedding import OverlapAwareSpeakerEmbedding
from .segmentation import SpeakerSegmentation
from .utils import Binarize
from .. import models as m
from .GMM_vad import measure_vad
from webrtcvad import Vad


class SpeakerDiarizationConfig(base.PipelineConfig):
    def __init__(
        self,
        segmentation: m.SegmentationModel | None = None,
        embedding: m.EmbeddingModel | None = None,
        duration: float = 5,
        step: float = 0.5,
        latency: float | Literal["max", "min"] | None = None,
        tau_active: float = 0.6,
        rho_update: float = 0.3,
        delta_new: float = 1,
        gamma: float = 3,
        beta: float = 10,
        max_speakers: int = 20,
        normalize_embedding_weights: bool = False,
        device: torch.device | None = None,
        sample_rate: int = 16000,
        clustering_timeout = None,
        vad_resolution = 4,
        vad_filtering=False,
        **kwargs,
    ):
        # Default segmentation model is pyannote/segmentation
        self.segmentation = segmentation or m.SegmentationModel.from_pyannote(
            "pyannote/segmentation"
        )

        # Default embedding model is pyannote/embedding
        self.embedding = embedding or m.EmbeddingModel.from_pyannote(
            "pyannote/embedding"
        )

        self._duration = duration
        self._sample_rate = sample_rate

        # Latency defaults to the step duration
        self._step = step
        self._latency = latency
        if self._latency is None or self._latency == "min":
            self._latency = self._step
        elif self._latency == "max":
            self._latency = self._duration

        self.tau_active = tau_active
        self.rho_update = rho_update
        self.delta_new = delta_new
        self.gamma = gamma
        self.beta = beta
        self.max_speakers = max_speakers
        self.normalize_embedding_weights = normalize_embedding_weights
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.clustering_timeout = clustering_timeout
        self.vad_resolution=vad_resolution
        self.vad_filtering=vad_filtering

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def step(self) -> float:
        return self._step

    @property
    def latency(self) -> float:
        return self._latency

    @property
    def sample_rate(self) -> int:
        return self._sample_rate


class SpeakerDiarization(base.Pipeline):
    def __init__(self, config: SpeakerDiarizationConfig | None = None,time_generator : Generator = None):
        self._config = SpeakerDiarizationConfig() if config is None else config
        
        #time generator informs the clustering module of the current time
        self.time_generator = time_generator
        
        msg = f"Latency should be in the range [{self._config.step}, {self._config.duration}]"
        assert self._config.step <= self._config.latency <= self._config.duration, msg

        self.segmentation = SpeakerSegmentation(
            self._config.segmentation, self._config.device
        )
        self.embedding = OverlapAwareSpeakerEmbedding(
            self._config.embedding,
            self._config.gamma,
            self._config.beta,
            norm=1,
            normalize_weights=self._config.normalize_embedding_weights,
            device=self._config.device,
        )
        self.pred_aggregation = DelayedAggregation(
            self._config.step,
            self._config.latency,
            strategy="hamming",
            cropping_mode="loose",
        )
        self.audio_aggregation = DelayedAggregation(
            self._config.step,
            self._config.latency,
            strategy="first",
            cropping_mode="center",
        )
        self.binarize = Binarize(self._config.tau_active)

        self.vad = Vad()
        
        # Internal state, handle with care
        self.timestamp_shift = 0
        self.clustering = None
        self.chunk_buffer, self.pred_buffer = [], []
        self.reset()

    @staticmethod
    def get_config_class() -> type:
        return SpeakerDiarizationConfig

    @staticmethod
    def suggest_metric() -> BaseMetric:
        return DiarizationErrorRate(collar=0, skip_overlap=False)

    @staticmethod
    def hyper_parameters() -> Sequence[base.HyperParameter]:
        return [base.TauActive, base.RhoUpdate, base.DeltaNew]

    @property
    def config(self) -> SpeakerDiarizationConfig:
        return self._config
    

    def set_timestamp_shift(self, shift: float):
        self.timestamp_shift = shift

    def reset(self):
        def default_time_generator():
            t = 0
            while True:
                yield t
                t+=self._config.step
                
        time_generator = self.time_generator or default_time_generator()
        
        self.set_timestamp_shift(0)
        self.clustering = OnlineSpeakerClustering(
            self.config.tau_active,
            self.config.rho_update,
            self.config.delta_new,
            "cosine",
            self.config.max_speakers,
            self.config.clustering_timeout or None,
            time_generator=time_generator
        )
        self.chunk_buffer, self.pred_buffer = [], []

    def apply_vad(self,wav:SlidingWindowFeature,resolution:float):
        #TODO figure out why this is so slow
       
        
        sr = self.config.sample_rate
        vad_seg_samples = int(np.floor(resolution * sr)) *2
        assert vad_seg_samples>=480, 'vad resolution too high!'
        
        #convert to pcm 16
        wav_pcm16 = (wav.data.copy().flatten()*2**15).astype(np.int16)
        wav_pcm16.resize((len(wav_pcm16)//vad_seg_samples,vad_seg_samples))
        vad = np.apply_along_axis(measure_vad,1,wav_pcm16, sr=sr,vad_object=self.vad,step_ms=5)
        
        binarized_vad = np.repeat(vad > self.config.tau_active,2)
        
        return binarized_vad
    
    def apply_vad_batch(self,wav:torch.Tensor,resolution:float):
        #TODO figure out why this is so slow
        
        sr = self.config.sample_rate
        vad_seg_samples = int(np.floor(resolution * sr)) *2
        assert vad_seg_samples>=480, 'vad resolution too high!'
        
        vad = []
        #convert to pcm 16
        wav_pcm16 = (wav*2**15).to(torch.int16)
        for batch in torch.unbind(wav_pcm16,dim=0):
            batch.resize_(batch.shape[0]//vad_seg_samples,vad_seg_samples)
            vad.append(torch.tensor([
                measure_vad(x_i,sr,self.vad,1) for x_i in torch.unbind(batch, dim=0)
            ]))
        vad = torch.stack(vad,dim=0)
        binarized_vad = torch.repeat_interleave(vad > self.config.tau_active,2,dim=1)
        return binarized_vad
        
    def __call__(
        self, waveforms: Sequence[SlidingWindowFeature]
    ) -> Sequence[tuple[Annotation, SlidingWindowFeature]]:
        """Diarize the next audio chunks of an audio stream.

        Parameters
        ----------
        waveforms: Sequence[SlidingWindowFeature]
            A sequence of consecutive audio chunks from an audio stream.

        Returns
        -------
        Sequence[tuple[Annotation, SlidingWindowFeature]]
            Speaker diarization of each chunk alongside their corresponding audio.
        """
        batch_size = len(waveforms)
        msg = "Pipeline expected at least 1 input"
        assert batch_size >= 1, msg

        # Create batch from chunk sequence, shape (batch, samples, channels)
        batch = torch.stack([torch.from_numpy(w.data) for w in waveforms])

        expected_num_samples = int(
            np.rint(self.config.duration * self.config.sample_rate)
        )
        msg = f"Expected {expected_num_samples} samples per chunk, but got {batch.shape[1]}"
        assert batch.shape[1] == expected_num_samples, msg

        # Extract segmentation and embeddings
        segmentations = self.segmentation(batch)  # shape (batch, frames, speakers)
                
        seg_resolution = waveforms[0].extent.duration / segmentations.shape[1]
        
        if self.config.vad_filtering:
            vad = self.apply_vad_batch(batch,seg_resolution)     
            vad = torch.repeat_interleave(vad.unsqueeze(2),segmentations.shape[-1],2)
            vad.resize_as_(segmentations)
            
            #filter segmentation based on vad results
            segmentations*=vad
        
        # embeddings has shape (batch, speakers, emb_dim)
        embeddings = self.embedding(batch, segmentations)

        
        outputs = []
        for wav, seg, emb in zip(waveforms, segmentations, embeddings):
            # Add timestamps to segmentation
            sw = SlidingWindow(
                start=wav.extent.start,
                duration=seg_resolution,
                step=seg_resolution,
            )
               
            seg = SlidingWindowFeature(seg.cpu().numpy(), sw)           
                        
            # Update clustering state and permute segmentation
            permuted_seg = self.clustering(seg, emb)

            # Update sliding buffer
            self.chunk_buffer.append(wav)
            self.pred_buffer.append(permuted_seg)

            # Aggregate buffer outputs for this time step
            agg_waveform = self.audio_aggregation(self.chunk_buffer)
            agg_prediction = self.pred_aggregation(self.pred_buffer)
            agg_prediction = self.binarize(agg_prediction)

            # Shift prediction timestamps if required
            if self.timestamp_shift != 0:
                shifted_agg_prediction = Annotation(agg_prediction.uri)
                for segment, track, speaker in agg_prediction.itertracks(
                    yield_label=True
                ):
                    new_segment = Segment(
                        segment.start + self.timestamp_shift,
                        segment.end + self.timestamp_shift,
                    )
                    shifted_agg_prediction[new_segment, track] = speaker
                agg_prediction = shifted_agg_prediction

            outputs.append((agg_prediction, agg_waveform))

            # Make place for new chunks in buffer if required
            if len(self.chunk_buffer) == self.pred_aggregation.num_overlapping_windows:
                self.chunk_buffer = self.chunk_buffer[1:]
                self.pred_buffer = self.pred_buffer[1:]

        return outputs
