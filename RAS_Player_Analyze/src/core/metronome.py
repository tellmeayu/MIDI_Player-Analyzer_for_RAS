import pygame
import numpy as np
from typing import Dict, Tuple

# Sound type constants
SOUND_TYPE_CLICK = "click"
SOUND_TYPE_BEEP = "beep"
SOUND_TYPE_WOOD = "wood"
SOUND_TYPE_TICK = "tick"

class LowLatencyAudioConfig:
    """Configuration for ultra-low latency audio"""
    
    # Ultra-low latency settings
    SAMPLE_RATE = 44100
    BUFFER_SIZE = 256  # Reduced from 1024 for lower latency
    CHANNELS = 2
    SAMPLE_SIZE = -16  # 16-bit signed
    
    # Sound generation parameters
    SOUND_DURATION = 0.08  # Very short duration (80ms) for minimal overlap
    ATTACK_TIME = 0.002   # 2ms attack for sharp transient
    RELEASE_TIME = 0.05   # 50ms release for natural decay
    
    # Pre-calculated constants
    ATTACK_SAMPLES = int(ATTACK_TIME * SAMPLE_RATE)
    RELEASE_SAMPLES = int(RELEASE_TIME * SAMPLE_RATE)
    TOTAL_SAMPLES = int(SOUND_DURATION * SAMPLE_RATE)

class SynthesizedSoundGenerator:
    """High-performance synthesized sound generator for metronome"""
    
    def __init__(self):
        """Initialize sound generator"""
        self.config = LowLatencyAudioConfig()
        self._sound_cache: Dict[str, bytes] = {}
        self._generate_all_sounds()
    
    def _generate_all_sounds(self):
        """Pre-generate all metronome sounds for instant playback"""
        
        # Generate different sound types
        self._sound_cache['click'] = self._generate_click_sound(800, 0.8)
        self._sound_cache['beep'] = self._generate_beep_sound(880, 0.6)
        self._sound_cache['wood'] = self._generate_wood_sound(200, 0.6)
        self._sound_cache['tick'] = self._generate_tick_sound(1200, 0.8)
            
    def _generate_click_sound(self, frequency: float, amplitude: float) -> bytes:
        """Generate sharp click sound with fast attack
        
        Args:
            frequency: Base frequency in Hz
            amplitude: Amplitude (0.0-1.0)
            
        Returns:
            bytes: Raw audio data
        """
        samples = self.config.TOTAL_SAMPLES
        t = np.linspace(0, self.config.SOUND_DURATION, samples)
        
        # Sharp click with harmonic content
        wave = np.sin(2 * np.pi * frequency * t) * 0.6
        wave += np.sin(2 * np.pi * frequency * 2 * t) * 0.3  # 2nd harmonic
        wave += np.sin(2 * np.pi * frequency * 3 * t) * 0.1  # 3rd harmonic
        
        # Very sharp attack envelope for percussive sound
        envelope = self._create_percussive_envelope(samples)
        wave = wave * envelope * amplitude
        
        return self._convert_to_audio_data(wave)
    
    def _generate_beep_sound(self, frequency: float, amplitude: float) -> bytes:
        """Generate pure sine wave beep
        
        Args:
            frequency: Frequency in Hz
            amplitude: Amplitude (0.0-1.0)
            
        Returns:
            bytes: Raw audio data
        """
        samples = self.config.TOTAL_SAMPLES
        t = np.linspace(0, self.config.SOUND_DURATION, samples)
        
        # Pure sine wave
        wave = np.sin(2 * np.pi * frequency * t)
        
        # Smooth envelope
        envelope = self._create_smooth_envelope(samples)
        wave = wave * envelope * amplitude
        
        return self._convert_to_audio_data(wave)
    
    def _generate_wood_sound(self, base_freq: float, amplitude: float) -> bytes:
        """Generate wooden block sound simulation
        
        Args:
            base_freq: Base frequency in Hz
            amplitude: Amplitude (0.0-1.0)
            
        Returns:
            bytes: Raw audio data
        """
        samples = self.config.TOTAL_SAMPLES
        t = np.linspace(0, self.config.SOUND_DURATION, samples)
        
        # Simulate wood block with filtered noise and resonance
        # Multiple resonant frequencies for wood-like timbre
        wave = np.sin(2 * np.pi * base_freq * t) * 0.4
        wave += np.sin(2 * np.pi * base_freq * 2.1 * t) * 0.3
        wave += np.sin(2 * np.pi * base_freq * 3.7 * t) * 0.2
        wave += np.sin(2 * np.pi * base_freq * 5.3 * t) * 0.1
        
        # Add controlled noise for texture
        noise = np.random.normal(0, 0.1, samples)
        wave += noise * 0.2
        
        # Very percussive envelope
        envelope = self._create_percussive_envelope(samples)
        wave = wave * envelope * amplitude
        
        return self._convert_to_audio_data(wave)
    
    def _generate_tick_sound(self, frequency: float, amplitude: float) -> bytes:
        """Generate high-frequency tick sound
        
        Args:
            frequency: Frequency in Hz
            amplitude: Amplitude (0.0-1.0)
            
        Returns:
            bytes: Raw audio data
        """
        samples = self.config.TOTAL_SAMPLES
        t = np.linspace(0, self.config.SOUND_DURATION, samples)
        
        # High-frequency tick with quick decay
        wave = np.sin(2 * np.pi * frequency * t) * 0.7
        wave += np.sin(2 * np.pi * frequency * 0.5 * t) * 0.3  # Sub-harmonic
        
        # Ultra-sharp envelope for tick effect
        envelope = self._create_tick_envelope(samples)
        wave = wave * envelope * amplitude
        
        return self._convert_to_audio_data(wave)
    
    def _create_percussive_envelope(self, samples: int) -> np.ndarray:
        """Create percussive envelope with sharp attack and exponential decay
        
        Args:
            samples: Number of samples
            
        Returns:
            np.ndarray: Envelope
        """
        envelope = np.ones(samples)
        
        # Sharp attack (2ms)
        if self.config.ATTACK_SAMPLES > 0:
            attack = np.linspace(0, 1, self.config.ATTACK_SAMPLES)
            envelope[:self.config.ATTACK_SAMPLES] = attack
        
        # Exponential decay for percussive sound
        if self.config.RELEASE_SAMPLES > 0:
            decay_start = samples - self.config.RELEASE_SAMPLES
            decay = np.exp(-np.linspace(0, 4, self.config.RELEASE_SAMPLES))  # e^(-4x) decay
            envelope[decay_start:] = decay
        
        return envelope
    
    def _create_smooth_envelope(self, samples: int) -> np.ndarray:
        """Create smooth envelope for sustained sounds
        
        Args:
            samples: Number of samples
            
        Returns:
            np.ndarray: Envelope
        """
        envelope = np.ones(samples)
        
        # Smooth attack
        if self.config.ATTACK_SAMPLES > 0:
            attack = np.sin(np.linspace(0, np.pi/2, self.config.ATTACK_SAMPLES))
            envelope[:self.config.ATTACK_SAMPLES] = attack
        
        # Smooth release
        if self.config.RELEASE_SAMPLES > 0:
            release_start = samples - self.config.RELEASE_SAMPLES
            release = np.sin(np.linspace(np.pi/2, 0, self.config.RELEASE_SAMPLES))
            envelope[release_start:] = release
        
        return envelope
    
    def _create_tick_envelope(self, samples: int) -> np.ndarray:
        """Create ultra-sharp envelope for tick sounds
        
        Args:
            samples: Number of samples
            
        Returns:
            np.ndarray: Envelope
        """
        envelope = np.zeros(samples)
        
        # Ultra-sharp attack (1ms)
        attack_samples = int(0.001 * self.config.SAMPLE_RATE)
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Very fast exponential decay
        decay_samples = samples - attack_samples
        if decay_samples > 0:
            decay = np.exp(-np.linspace(0, 6, decay_samples))  # e^(-6x) very fast decay
            envelope[attack_samples:] = decay
        
        return envelope
    
    def _convert_to_audio_data(self, wave: np.ndarray) -> bytes:
        """Convert numpy array to audio data
        
        Args:
            wave: Wave data as numpy array
            
        Returns:
            bytes: Raw audio data
        """
        # Normalize and convert to 16-bit PCM
        wave = np.clip(wave, -1.0, 1.0)
        wave_int16 = (wave * 32767).astype(np.int16)
        
        # Convert to stereo
        if self.config.CHANNELS == 2:
            stereo_data = np.column_stack((wave_int16, wave_int16))
            return stereo_data.tobytes()
        else:
            return wave_int16.tobytes()
    
    def get_sound_data(self, sound_type: str) -> bytes:
        """Get pre-generated sound data
        
        Args:
            sound_type: Type of sound (click, beep, wood, tick)
            
        Returns:
            bytes: Raw audio data
        """
        key = f"{sound_type}"
        return self._sound_cache.get(key, self._sound_cache['click'])
    
class BaseMetronome:
    """Base metronome class with low latency synthesized audio"""
    
    def __init__(self):
        """Initialize low-latency metronome"""
        self.is_active = False
        self.beat_count = 4
        self.beat_denominator = 4
        self.volume = 1.0
        self.sound_type = SOUND_TYPE_CLICK
        
        # Initialize low latency audio
        self._init_low_latency_audio()
        
        # Initialize synthesized sound generator
        self.sound_generator = SynthesizedSoundGenerator()
        
        # Create pygame Sound objects from synthesized data
        self._create_sound_objects()
    
    def _init_low_latency_audio(self):
        """Initialize pygame mixer for ultra-low latency"""
        config = LowLatencyAudioConfig()
        
        try:
            # Stop any existing mixer
            if pygame.mixer.get_init():
                pygame.mixer.quit()
            
            # Initialize with low-latency settings
            pygame.mixer.pre_init(
                frequency=config.SAMPLE_RATE,
                size=config.SAMPLE_SIZE,
                channels=config.CHANNELS,
                buffer=config.BUFFER_SIZE  # Small buffer for low latency
            )
            pygame.mixer.init()
            
        except Exception as e:
            print(f"Error initializing low-latency audio: {e}")
            # Fallback to default settings
            pygame.mixer.init()
    
    def _create_sound_objects(self):
        """Create pygame Sound objects from synthesized data"""
        try:
            # Get sound data for current sound type
            sound_data = self.sound_generator.get_sound_data(self.sound_type)
            
            # Create Sound object
            self.beat_sound = pygame.mixer.Sound(buffer=sound_data)
            
            # Set volume
            self.beat_sound.set_volume(self.volume)
                        
        except Exception as e:
            print(f"Error creating sound objects: {e}")
            # Fallback to simple generated sounds
            self._generate_fallback_sounds()
    
    def _generate_fallback_sounds(self):
        """Generate simple fallback sounds if main generation fails"""
        try:
            # Simple sine wave generation
            sample_rate = 44100
            duration = 0.1
            
            # Generate simple tone
            strong_freq = 880  # A5
            sound_data = self._generate_simple_tone(strong_freq, duration, sample_rate, 0.8)
            
            self.beat_sound = pygame.mixer.Sound(buffer=sound_data)
            
            print("Using fallback metronome sounds")
            
        except Exception as e:
            print(f"Error generating fallback sounds: {e}")
    
    def _generate_simple_tone(self, frequency: float, duration: float, sample_rate: int, amplitude: float) -> bytes:
        """Generate simple sine wave tone
        
        Args:
            frequency: Frequency in Hz
            duration: Duration in seconds
            sample_rate: Sample rate
            amplitude: Amplitude (0.0-1.0)
            
        Returns:
            bytes: Raw audio data
        """
        samples = int(duration * sample_rate)
        t = np.linspace(0, duration, samples)
        wave = np.sin(2 * np.pi * frequency * t) * amplitude
        
        # Simple envelope
        envelope = np.ones(samples)
        attack = int(0.01 * sample_rate)
        release = int(0.05 * sample_rate)
        
        if attack > 0:
            envelope[:attack] = np.linspace(0, 1, attack)
        if release > 0:
            envelope[-release:] = np.linspace(1, 0, release)
        
        wave = wave * envelope
        
        # Convert to stereo 16-bit
        wave_int16 = (wave * 32767).astype(np.int16)
        stereo_data = np.column_stack((wave_int16, wave_int16))
        
        return stereo_data.tobytes()
    
    def set_beat(self, count, denominator=4):
        """Set time signature
        
        Args:
            count: Beats per measure
            denominator: Note value per beat
        """
        self.beat_count = max(1, min(12, count))
        self.beat_denominator = denominator
    
    def set_volume(self, volume):
        """Set metronome volume
        
        Args:
            volume: Volume value, range 0.0-1.0
        """
        self.volume = max(0.0, min(1.0, volume))
        
        if hasattr(self, 'beat_sound'):
            self.beat_sound.set_volume(self.volume)
            
    def set_sound_type(self, sound_type):
        """Set metronome sound type
        
        Args:
            sound_type: Sound type (click, beep, wood, tick)
        """
        if sound_type in [SOUND_TYPE_CLICK, SOUND_TYPE_BEEP, SOUND_TYPE_WOOD, SOUND_TYPE_TICK]:
            self.sound_type = sound_type
            self._create_sound_objects()
        else:
            print(f"Warning: Unknown sound type '{sound_type}', keeping current type")
    
    def get_latency_info(self) -> Dict[str, float]:
        """Get latency information for debugging
        
        Returns:
            Dict: Latency information
        """
        config = LowLatencyAudioConfig()
        
        # Calculate theoretical latency
        buffer_latency = config.BUFFER_SIZE / config.SAMPLE_RATE * 1000  # ms
        
        return {
            'buffer_size': config.BUFFER_SIZE,
            'sample_rate': config.SAMPLE_RATE,
            'theoretical_latency_ms': buffer_latency,
            'sound_duration_ms': config.SOUND_DURATION * 1000,
            'attack_time_ms': config.ATTACK_TIME * 1000
        }