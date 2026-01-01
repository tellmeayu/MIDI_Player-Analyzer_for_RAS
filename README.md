**Project Description**
<table>
  <tr>
    <td style="width: 33.3%;">
      <img src="https://www.sylviastudio.cn/wp-content/uploads/2025/11/Screenshot-2025-11-23-at-19.21.57.png" alt="4-dim framework demo" style="width: 100%; border-radius: 8px;">
      <br>
      <b>4-dim analysis visualization</b>
    </td>
    <td style="width: 33.3%;">
      <img src="https://www.sylviastudio.cn/wp-content/uploads/2025/11/Screenshot-2025-11-23-at-16.55.43.png" alt="Batch Processing" style="width: 100%; border-radius: 8px;">
      <br>
      <b>Batch Processing window</b>
    </td>
    <td style="width: 33.3%;">
      <img src="https://www.sylviastudio.cn/wp-content/uploads/2025/11/Screenshot-2025-11-23-at-21.14.09.png" alt="Radar Chart" style="width: 100%; border-radius: 8px;">
      <br>
      <b>main player window</b>
    </td>
  </tr>
</table>
This project is a specialized platform designed to support clinicians working with Rhythmic Auditory Stimulation (RAS) therapy for gait rehabilitation. RAS is mainly applied as an alternative rehab method for patients with Parkinson's disease, stroke, and other movement disorders. At its heart, the system tackles a fundamental challenge in music therapy: identifying which pieces of music are actually suitable for therapeutic use. While any MIDI player can produce sound, this platform goes deeper by analyzing the rhythmic structure of music across multiple dimensions, including beat density, predictability, accent patterns, and rhythmic uniformity. This is expected to help therapists understand whether a piece will support or hinder motor entrainment. To categorize music into clinically meaningful groups (duple vs. triple meter), a dual-layer meter classification system was designed. It uses both MIDI pattern analysis and neural network-based audio processing (RNN+DBN). Beyond analysis, the platform provides microsecond-accurate playback with an integrated therapy metronome that can be precisely synchronized to music, supporting cadences from 20 to 180 for gait training. Smart timing correction is another small tool which handles common MIDI quantization error (shifting from meter grid). The interface remains deliberately simple—resembling a straightforward MIDI player—but surfaces sophisticated features only when needed.

**Key Features**

- **High-Precision Playback**: Microsecond-accurate MIDI synthesis with FluidSynth and real-time metronome synchronization
- **4D Rhythm Analysis**: Quantifies musical rhythm across Beat Density, Predictability, Beat Salience, and Rhythmic Uniformity dimensions
- **Smart Timing Correction**: Pattern-based downbeat detection with neural network validation and MIDI event adjustment
- **Batch Processing**: Process-isolated analysis with dual-layer meter classification (MIDI + audio) and persistent caching
- **RAS Therapy Tools**: Direct cadence control (20-180 steps/min), unified downbeat cueing, and clinical gait training optimization
- **Advanced Architecture**: SQLite caching, atomic IPC, CLI automation, and cross-platform compatibility

Full description (with algorithm explanation that utilized in 4-dim framework): https://www.sylviastudio.cn/overview-ras-player/

---

### Playback basic
<figure>
<img src="https://www.sylviastudio.cn/wp-content/uploads/2025/11/Screenshot-2025-11-23-at-21.14.09.png" width="500">
</figure>

It just looks like a simple MIDI player plus a tempo/cadence control! I designed the front-end to be clean and intuitive, focusing purely on the basic operation. All the heavy lifting is working silently behind the scenes, only surfacing data when specifically requested or automatically detected. I deliberately kept the UI minimal. 

Once you open a MIDI:
<figure>
<img src="https://www.sylviastudio.cn/wp-content/uploads/2025/11/playback-system.png" width="500">
</figure>

If a loaded MIDI lacks sufficient metadata (e.g. Type 0), the system automatically enter an assisted analysis mode:
<figure><img src="https://www.sylviastudio.cn/wp-content/uploads/2025/11/type0-mode.png" width="450">
</figure>

<figure><img src="https://www.sylviastudio.cn/wp-content/uploads/2025/11/beat-regularity.png" width="450"></figure>

An additional audio player is provided, which features the music mixed with a precisely synchronized metronome cue for training purposes. More importantly, once the beat tracking finished, system will immediately calculate **Beat Regularity**, measuring if it's suitable for RAS gait training. The standard set here is rather strict considering the training purpose.

If a standard MIDI exhibits slight quantization shift that prevents metronome alignment, the "Timing Correction" tool is engaged. Once you applied the correction:
<figure><img src="https://www.sylviastudio.cn/wp-content/uploads/2025/11/Screenshot-2025-11-23-at-17.04.26-e1763904871842.png" width="450"></figure>
It calculates the optimal downbeat shift and  applies it automatically. However, given the abstract and complex nature of musical rhythm, the algorithm output isn't always precise. Therefore, a manual fine-tuning controls are provided for human validation.

### Batch Analysis for Meter Categorization
The goal of meter estimation here is not to "guess" the time signature, but to robustly categorize the music into "duple" or "triple" classes based on its perceived "meter feeling" (critical for RAS). There are two analysis modes. The fast mode performs analysis purely on MIDI data without giving meter results. The Deep Mode, conversely, provides a probabilistic meter category, though only for reference. The algorithm uses a "validation strategy" to iterate possibilities. For macro-structural analysis, the system currently limits its scope to the most common regular meter types (2/4, 3/4, 4/4), focusing on macro-level duple/triple distinction rather than full time-signature identification.
<figure><img src="https://www.sylviastudio.cn/wp-content/uploads/2025/11/Screenshot-2025-11-23-at-16.55.43.png" width="450"></figure>
<figure><img src="https://www.sylviastudio.cn/wp-content/uploads/2025/11/Screenshot-2025-11-23-at-16.56.15.png" width="450"></figure>
<figure><img src="https://www.sylviastudio.cn/wp-content/uploads/2025/11/deep-mode-results-1.png" width="450"></figure>

### 4-dim Rhythm Analyzer with Radar Chart Visualization

The 4-D analysis result is visualized on a Radar Chart (with randomly assigned colors). This visualization allows for immediate, comparative assessment of the music's rhythmic profile.
For instance, Debussy's *Clair de Lune* serves as an extreme example. Its low Predictability score suggests that the rhythm is likely too ambiguous for a listener to reliably tap or walk along with:
<figure><img src="https://www.sylviastudio.cn/wp-content/uploads/2025/11/Screenshot-2025-11-25-at-15.53.53.png" width="400"></figure>

A Scarlatti keyboard sonata may also be difficult to follow, but due to a different profile, such as extremely low Beat Salience (a "flat" or non-accented rhythmic feel):
<figure><img src="https://www.sylviastudio.cn/wp-content/uploads/2025/11/Screenshot-2025-11-23-at-19.07.56.png" width="400"></figure>

On the contrary, some music sounds quite "on-beat", that usually indicates a good candidate used for RAS gait training:
<figure><img src="https://www.sylviastudio.cn/wp-content/uploads/2025/11/Screenshot-2025-11-25-at-15.52.08.png" width="400"></figure>
<figure><img src="https://www.sylviastudio.cn/wp-content/uploads/2025/11/Screenshot-2025-11-25-at-15.51.28.png" width="400"></figure>

Low note density is also a critical factor need to be considered, especially when matched with a patient's low cadence. Insufficient rhythmic information can lead to **perceptive or cognitive processing failure** rather than rhythmic entrainment.

Conversely, music with high speed and high density provides information that is too cognitively "dense" to process. This complexity can decrease the patient's perceived "joyfulness" or impair the brain's ability to encode the signals into a coherent "musical line".

This model, while effective as a first stage, is far from perfect. I believe there is significant room for improvement across its feature set, underlying algorithms, and dimensionality.

---

## Architecture

The system implements a modular, process-isolated architecture. 
```
Core Folder Structure
src/
├── main.py                           # Application entry point
├── __init__.py                       # Package initialization
├── core/                             # Core playback engine
│   ├── __init__.py
│   ├── midi_engine.py                # MIDI synthesis and playback engine
│   ├── precision_timer.py            # High-precision timing system
│   ├── event_scheduler.py            # Real-time event scheduling
│   ├── metronome.py                  # Audio metronome implementation
│   ├── ras_therapy_metronome.py      # RAS-optimized metronome
│   ├── player_session_state.py       # Session state management
│   ├── playback_mode.py              # Playback mode definitions
│   ├── beat_timeline.py              # Beat position management
│   ├── track_activity_monitor.py     # Track visualization data
│   ├── midi_metadata.py              # MIDI metadata handling
│   ├── gm_instruments.py             # General MIDI instrument definitions
│   └── audio_cache.py                # Audio file caching system
├── analysis/                         # Musical analysis toolkit
│   ├── __init__.py
│   ├── anacrusis_detector.py         # First downbeat detection
│   ├── beat_tracker_basic.py         # Basic beat tracking
│   ├── beat_tracking_service.py      # Beat tracking service layer
│   ├── preprocessor.py               # MIDI preprocessing utilities
├── batch_filter/                     # Batch processing system
│   ├── __init__.py
│   ├── core/                         # Core analysis algorithms
│   │   ├── __init__.py
│   │   ├── batch_processor.py        # Multi-stage batch processor
│   │   ├── tempogram_analyzer.py     # MIDI-based rhythm analysis
│   │   └── meter_estimator.py        # Audio-based meter estimation
│   ├── cli/                          # Command-line interface
│   │   ├── __init__.py
│   │   └── batch_analyzer_cli.py     # CLI subprocess implementation
│   ├── ui/                           # Batch analysis UI
│   │   ├── __init__.py
│   │   └── batch_analyzer_window.py  # Independent analysis window
│   └── cache/                        # Caching layer
│       ├── __init__.py
│       └── library_manager.py        # SQLite-based result caching
├── multi_dim_analyzer/               # 4D rhythm analysis framework
│   ├── __init__.py
│   ├── config.py                     # Analysis configuration
│   ├── pipeline.py                   # Main analysis orchestration
│   ├── plotting.py                   # Visualization utilities
│   ├── beat_density.py               # Dimension I: Beat density analysis
│   ├── predictability.py             # Dimension II: Metrical predictability
│   ├── beat_salience.py              # Dimension III: Beat salience
│   ├── rhythmic_uniformity.py        # Dimension IV: Rhythmic uniformity
│   └── utils/                        # Analysis utilities
│       ├── __init__.py
│       ├── beat_grid.py              # Beat grid generation
│       └── midi_processor.py         # MIDI processing utilities
└── ui/                               # User interface components
    ├── __init__.py
    ├── gui.py                        # Main application window
    ├── playback_controls.py          # Playback control widgets
    ├── track_visualization.py        # Real-time track visualization
    ├── dialogs.py                    # Dialog windows
    ├── analysis_dialogs.py           # Analysis-related dialogs
    ├── menu_manager.py               # Menu system management
    ├── utilities.py                  # UI utility functions
    ├── file_info_display.py          # File information display
    ├── anacrusis_tool_window.py      # Timing correction tool window
    ├── rhythm_analysis_dialog.py     # Rhythm analysis interface
    ├── rhythm_analysis_worker.py     # Analysis worker thread
    ├── audio_player_launcher.py      # Audio player launch utilities
    ├── audio_player_window.py        # Audio player interface
    └── beat_tracking_worker.py       # Beat tracking background worker
```
---

***Author's notes*** 
*This project is a refined successor to an ambitious early concept. The initial motivation came from my basic training in Neurologic Music Therapy last year(2024). My first idea was to simulate a complete RAS therapy session, including gait detection via cutting-edge computer vision technology like MediaPipe Pose (https://github.com/tellmeayu/RAS-helper.git). However, given the technical complexity and resource limitation of a solo developer, I made the strategic decision to narrow down the scope. This allows me to focus on my own major and allocate all development efforts toward mastering the platform's rhythm analysis core.*
