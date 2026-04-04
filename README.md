# Computer-Assisted Music Analysis and Screening for Rhythmic Auditory Stimulation

A research-oriented music analysis and playback system for Rhythmic Auditory Stimulation (RAS) in gait rehabilitation.

This project combines MIDI-centered playback control, interpretable rhythmic feature analysis, and batch meter screening to support music selection in rehabilitation scenarios. It was developed as part of my master's thesis in Music Technology, with a focus on digital music intelligent processing.

## Overview

Rhythmic Auditory Stimulation (RAS) is a well-established neurorehabilitation technique for improving gait impairments. However, in real rehabilitation practice, music selection is still highly dependent on therapist experience. Existing workflows often rely on repeated manual listening, metronome comparison, or general-purpose audio software, which makes music screening inefficient and difficult to standardize.

This project addresses that gap by translating experience-based judgments of rhythmic suitability into interpretable computational indicators and screening methods. The result is a desktop system that integrates:

- precise MIDI playback for rehabilitation use
- interpretable rhythmic feature analysis for single-track assessment
- batch meter screening for large music libraries

## Why This Project Matters

The project was designed for a specific application problem rather than as a general music player or a generic MIR toolkit.

In RAS-based gait rehabilitation, therapists often need music that is:

- rhythmically stable
- easy to synchronize with
- clear in beat structure
- suitable for controlled tempo adjustment

Despite the clinical relevance of these properties, there is still a lack of practical tools that can evaluate them in a structured and computationally interpretable way. This project aims to provide such a tool.

## Main Contributions

The study makes three main contributions:

- It proposes an interpretable four-dimensional rhythmic feature framework for assessing music suitability in RAS scenarios.
- It implements a dual-mode meter screening pipeline that combines symbolic MIDI analysis with audio-assisted analysis.
- It integrates playback, feature analysis, and batch screening into a unified desktop application.

## Core Features

- MIDI-based playback with tempo control adapted to gait training scenarios
- independent metronome control
- track-level solo and mute control for multi-track MIDI files
- timing correction support for slightly misaligned MIDI files
- four-dimensional rhythmic feature analysis with radar-chart visualization
- batch meter screening for music libraries
- fast symbolic mode for efficient large-scale screening
- deep audio-assisted mode for more complex metrical structures

## The Four-Dimensional Rhythmic Framework

A central part of this project is an interpretable framework for rhythmic suitability analysis in RAS. It models rhythm through four dimensions.

### 1. Uniformity

Uniformity measures the temporal stability of rhythmic events.

It reflects whether the underlying rhythmic intervals provide a stable reference for beat induction and sensorimotor synchronization. In this project, uniformity is derived from the variability of inter-onset intervals (IOIs) and mapped into standardized score.

### 2. Salience

Salience measures how clearly beat positions stand out from their local background.

A rhythm with higher beat salience provides clearer auditory anchors for synchronization. This dimension is intended to capture how strongly beat-relevant events emerge in the surface structure of the music.

### 3. Predictability

Predictability measures how well rhythmic events support beat expectation.

This dimension combines two aspects:

- macro-level beat coverage, which reflects whether expected beat locations are sufficiently reinforced
- micro-level metrical alignment, which reflects whether events align well with the internal strong-weak metrical structure

### 4. Density

Density measures the event load per beat.

This dimension is meant to reflect attentional load. Rhythms that are too sparse may fail to support stable beat perception, while rhythms that are too dense may overload the listener and reduce synchronization efficiency.

## Why an Interpretable Framework

The goal of this framework is not to replace therapists or to reduce music choice to a single automatic score. Instead, it is designed as a decision-support tool.

Compared with fully black-box prediction pipelines, this framework offers:

- interpretability at the feature level
- explicit links to rhythm perception theory
- clearer communication with domain experts
- a more transparent basis for future refinement and validation

## Dual-Mode Meter Screening

To support large-scale music library screening, the system includes a dual-mode meter classification pipeline.

### Fast Mode

Fast mode operates directly on MIDI data.

It performs efficient preliminary screening using symbolic rhythmic evidence, including tempo-domain periodicity analysis and time-domain verification. This mode is intended for large libraries and rapid first-pass filtering.

### Deep Mode

Deep mode supplements symbolic analysis with audio-based beat and downbeat inference.

It is designed for more complex cases in which symbolic information alone may fail to capture metrical structure reliably, especially in music with richer surface variation or ambiguous compound meter behavior.

### Why Two Modes

The two modes serve different practical needs:

- Fast mode prioritizes efficiency and scalability
- Deep mode prioritizes robustness in more complex samples

This design reflects a practical tradeoff between speed and precision in rehabilitation-oriented music screening workflows.

## User Interface

The system includes three main interface components:

- main playback interface for loading, controlling, and auditioning MIDI files
- batch screening interface for library-level meter classification
- rhythmic analysis interface for visualizing the four-dimensional profile of the currently loaded track

The UI is designed for practical use rather than only algorithm demonstration. It emphasizes clear interaction, low cognitive load, and direct support for therapist-oriented workflows.

## Example Use Cases

This system is intended to support tasks such as:

- screening a large MIDI collection for duple-oriented gait-training candidates
- analyzing whether a specific piece is rhythmically too dense or insufficiently salient
- comparing tracks not only by BPM or nominal time signature, but also by interpretable rhythmic characteristics
- adjusting playback tempo without pitch distortion using MIDI
- isolating specific tracks in multi-track MIDI files to reduce listening overload

## Technical Stack
This project is implemented primarily in Python.        
Main tools and libraries include:
- Python
- PyQt5 for the desktop interface
- Mido for MIDI parsing and message handling
- Librosa for audio analysis support
- Madmom for beat and downbeat related processing
- standard scientific Python tools for numerical computation and visualization

---

## User Interface
### Main Window
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
The goal of meter estimation here is not to "guess" the time signature, but to robustly categorize the music into "duple" or "triple" types based on its perceived "meter feeling" (critical for RAS). 
<figure><img src="https://www.sylviastudio.cn/wp-content/uploads/2025/11/Screenshot-2025-11-23-at-16.55.43.png" width="450"></figure>
<figure><img src="https://www.sylviastudio.cn/wp-content/uploads/2025/11/Screenshot-2025-11-23-at-16.56.15.png" width="450"></figure>
<figure><img src="https://www.sylviastudio.cn/wp-content/uploads/2025/11/deep-mode-results-1.png" width="450"></figure>

### 4-dim Rhythm Analyzer with Radar Chart Visualization

The 4-D analysis result is visualized on a Radar Chart (with randomly assigned colors). This visualization allows for immediate, comparative assessment of the music's rhythmic profile.
<table>
  <tr>
    <td style="width: 50%;">
      <img src="https://www.sylviastudio.cn/wp-content/uploads/2025/11/Screenshot-2025-11-23-at-19.21.57.png" alt="4-dim framework demo" style="width: 100%; border-radius: 8px;">
      <br>
      <b>4-dim case</b>
    </td>
    <td style="width: 50%;">
      <img src="https://www.sylviastudio.cn/wp-content/uploads/2026/02/Debussy.png" alt="4-dim framework demo" style="width: 100%; border-radius: 8px;">
      <br>
      <b>4-dim case</b>
    </td>
  </tr>
</table>

---

## Important Notes
- This repository contains a research prototype rather than a fully validated clinical product.

- The system is intended as a decision-support tool, not a clinical decision-making system.

- The four-dimensional framework is theory-informed and engineering-driven, but not yet fully validated through expert studies.

- Compound meter remains a challenging case because metrical structure can be interpreted differently depending on hierarchical level and analytical perspective.

- Future work should include expert annotation, user studies, and more systematic data-driven refinement.

## Relation to Thesis
This repository is based on my master's thesis in Music Technology.

Thesis topic: 
**MIDI Music Classification for RASTherapy: A Study and Prototype System Design**

Language of thesis: Chinese

English summary or translated materials: available upon request
