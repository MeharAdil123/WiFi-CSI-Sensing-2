"""
Generates the Master Presentation (WiFi_CSI_HAR_Final_Presentation.pptx).

Structured for 4 speakers (Group 10, Dept of ECE, RNSIT 2026-27):
- Speaker 1: Title, Motivation, Problem Statement, CSI Physics & Hardware
- Speaker 2: CSI Acquisition, Preprocessing Pipeline, Windowing & Data Audit
- Speaker 3: Model Evolution, CNN-BiLSTM, Proposed MS-CNN-BiGRU & SE-TCN
- Speaker 4: Leakage Diagnosis, Benchmark Results, Literature Comparison, Live Demo, Conclusion
"""

import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

# Design System Colors
C_DARK_BG   = RGBColor(15, 23, 42)      # #0F172A Slate 900
C_CARD_BG   = RGBColor(255, 255, 255)  # White
C_LIGHT_BG  = RGBColor(248, 250, 252)  # #F8FAFC Slate 50
C_PRIMARY   = RGBColor(13, 148, 136)   # #0D9488 Teal 600
C_SECONDARY = RGBColor(30, 41, 59)     # #1E293B Slate 800
C_TEXT_DARK = RGBColor(15, 23, 42)     # #0F172A Slate 900
C_TEXT_MUTED= RGBColor(100, 116, 139)  # #64748B Slate 500
C_ACCENT    = RGBColor(16, 185, 129)   # #10B981 Emerald 500
C_ALERT     = RGBColor(225, 29, 72)    # #E11D48 Rose 600
C_CARD_BORDER = RGBColor(226, 232, 240)# #E2E8F0 Slate 200


def create_deck():
    prs = pptx.Presentation()
    # 16:9 widescreen layout
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def add_header(slide, title_text, category_text, speaker_text):
        """Creates a standardized modern header on every content slide."""
        # Top banner background
        top_box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.1))
        top_box.fill.solid()
        top_box.fill.fore_color.rgb = C_DARK_BG
        top_box.line.color.rgb = C_DARK_BG

        # Category / Section pill
        cat_p = slide.shapes.add_textbox(Inches(0.8), Inches(0.12), Inches(8), Inches(0.3))
        tf = cat_p.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = category_text.upper()
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = C_PRIMARY

        # Slide Title
        title_p = slide.shapes.add_textbox(Inches(0.8), Inches(0.38), Inches(8.5), Inches(0.6))
        tf2 = title_p.text_frame
        tf2.word_wrap = True
        p2 = tf2.paragraphs[0]
        p2.text = title_text
        p2.font.size = Pt(22)
        p2.font.bold = True
        p2.font.color.rgb = RGBColor(255, 255, 255)

        # Speaker Badge
        badge = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(10.2), Inches(0.3), Inches(2.3), Inches(0.5))
        badge.fill.solid()
        badge.fill.fore_color.rgb = C_SECONDARY
        badge.line.color.rgb = C_PRIMARY
        badge.line.width = Pt(1.5)
        btf = badge.text_frame
        bp = btf.paragraphs[0]
        bp.text = speaker_text
        bp.font.size = Pt(11)
        bp.font.bold = True
        bp.font.color.rgb = RGBColor(255, 255, 255)
        bp.alignment = PP_ALIGN.CENTER

    def add_card(slide, left, top, width, height, title="", border_color=C_CARD_BORDER, bg_color=C_CARD_BG):
        """Creates a clean styled container card."""
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        card.line.color.rgb = border_color
        card.line.width = Pt(1)

        if title:
            tb = slide.shapes.add_textbox(Inches(left + 0.2), Inches(top + 0.15), Inches(width - 0.4), Inches(0.4))
            tf = tb.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.text = title
            p.font.size = Pt(13)
            p.font.bold = True
            p.font.color.rgb = C_SECONDARY
        return card

    # =========================================================================
    # SLIDE 1: TITLE SLIDE
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = C_DARK_BG
    bg1.line.color.rgb = C_DARK_BG

    # Sub-header: Institution
    inst_tb = s1.shapes.add_textbox(Inches(1.0), Inches(0.8), Inches(11.3), Inches(0.5))
    itf = inst_tb.text_frame
    ip = itf.paragraphs[0]
    ip.text = "RNS INSTITUTE OF TECHNOLOGY, BENGALURU — DEPARTMENT OF ECE"
    ip.font.size = Pt(13)
    ip.font.bold = True
    ip.font.color.rgb = C_PRIMARY

    # Main Project Title
    t_tb = s1.shapes.add_textbox(Inches(1.0), Inches(1.3), Inches(11.3), Inches(1.6))
    ttf = t_tb.text_frame
    ttf.word_wrap = True
    tp = ttf.paragraphs[0]
    tp.text = "Privacy-Preserving Through-Wall Human Activity Recognition Using WiFi-CSI Without Cameras"
    tp.font.size = Pt(32)
    tp.font.bold = True
    tp.font.color.rgb = RGBColor(255, 255, 255)

    sub_p = ttf.add_paragraph()
    sub_p.text = "ESP32-S3 Commodity Hardware • 5-Class Deep Learning • Multi-Seed Evaluation • Eldercare Fall Detection"
    sub_p.font.size = Pt(16)
    sub_p.font.color.rgb = C_TEXT_MUTED

    # Team Members Card
    team_card = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(3.4), Inches(6.8), Inches(3.2))
    team_card.fill.solid()
    team_card.fill.fore_color.rgb = C_SECONDARY
    team_card.line.color.rgb = C_PRIMARY
    tt_tb = s1.shapes.add_textbox(Inches(1.3), Inches(3.6), Inches(6.2), Inches(2.8))
    ttf = tt_tb.text_frame
    p = ttf.paragraphs[0]
    p.text = "PROJECT TEAM (GROUP 10) — 2026-2027"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = C_PRIMARY

    members = [
        ("01", "Md Mehar Adil D H", "1RN23EC094"),
        ("02", "Mohith Gowda M", "1RN23EC095"),
        ("03", "Nihar Gowda", "1RN23EC106"),
        ("04", "Sujal L G", "1RN23EC169")
    ]
    for num, name, usn in members:
        mp = ttf.add_paragraph()
        mp.text = f"• {name}  [{usn}]"
        mp.font.size = Pt(14)
        mp.font.color.rgb = RGBColor(255, 255, 255)

    # Guidance & Reviewer Card
    guide_card = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.2), Inches(3.4), Inches(4.1), Inches(3.2))
    guide_card.fill.solid()
    guide_card.fill.fore_color.rgb = C_SECONDARY
    guide_card.line.color.rgb = C_CARD_BORDER
    gt_tb = s1.shapes.add_textbox(Inches(8.4), Inches(3.6), Inches(3.7), Inches(2.8))
    gtf = gt_tb.text_frame
    gp = gtf.paragraphs[0]
    gp.text = "ACADEMIC ADVISORY"
    gp.font.size = Pt(13)
    gp.font.bold = True
    gp.font.color.rgb = C_PRIMARY

    p1 = gtf.add_paragraph()
    p1.text = "\nProject Guide:"
    p1.font.size = Pt(12)
    p1.font.color.rgb = C_TEXT_MUTED
    p2 = gtf.add_paragraph()
    p2.text = "Dr. Leena Chandrashekar"
    p2.font.size = Pt(14)
    p2.font.bold = True
    p2.font.color.rgb = RGBColor(255, 255, 255)
    p2_sub = gtf.add_paragraph()
    p2_sub.text = "Associate Professor, Dept. of ECE"
    p2_sub.font.size = Pt(11)
    p2_sub.font.color.rgb = C_TEXT_MUTED

    p3 = gtf.add_paragraph()
    p3.text = "\nProject Reviewer:"
    p3.font.size = Pt(12)
    p3.font.color.rgb = C_TEXT_MUTED
    p4 = gtf.add_paragraph()
    p4.text = "Dr. Srikanth G N"
    p4.font.size = Pt(14)
    p4.font.bold = True
    p4.font.color.rgb = RGBColor(255, 255, 255)
    p4_sub = gtf.add_paragraph()
    p4_sub.text = "Associate Professor, Dept. of ECE"
    p4_sub.font.size = Pt(11)
    p4_sub.font.color.rgb = C_TEXT_MUTED

    # =========================================================================
    # SLIDE 2: MOTIVATION & PROBLEM STATEMENT (Speaker 1)
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Motivation & Problem Statement: The Case for WiFi CSI", "Introduction & Context", "Speaker 1")

    # Card 1: Camera Limitations
    c1 = add_card(s2, 0.8, 1.4, 3.6, 5.5, "1. Camera-Based Sensing")
    tb1 = s2.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(3.2), Inches(4.5))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "Major Drawbacks:"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = C_ALERT
    bullets1 = [
        "Severe privacy invasion in private living spaces (bedrooms, restrooms).",
        "Strict optical Line-of-Sight (LOS) requirement.",
        "Completely blocked by opaque partitions, walls, and smoke.",
        "Fails in darkness or harsh glare.",
        "High storage and computational bandwidth for continuous video streams."
    ]
    for b in bullets1:
        bp = tf1.add_paragraph()
        bp.text = f"• {b}"
        bp.font.size = Pt(11)
        bp.font.color.rgb = C_TEXT_DARK

    # Card 2: Wearables Limitations
    c2 = add_card(s2, 4.8, 1.4, 3.6, 5.5, "2. Wearable-Based Sensing")
    tb2 = s2.shapes.add_textbox(Inches(5.0), Inches(2.0), Inches(3.2), Inches(4.5))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "Major Drawbacks:"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = C_ALERT
    bullets2 = [
        "Requires active user compliance and daily recharging.",
        "Older adults and dementia patients frequently forget or refuse devices.",
        "Removed during sleep or bathing (where over 60% of senior falls occur!).",
        "Skin irritation and discomfort with prolonged attachment.",
        "Costly multi-device replacements when lost or broken."
    ]
    for b in bullets2:
        bp = tf2.add_paragraph()
        bp.text = f"• {b}"
        bp.font.size = Pt(11)
        bp.font.color.rgb = C_TEXT_DARK

    # Card 3: Our Proposed WiFi Solution
    c3 = add_card(s2, 8.8, 1.4, 3.7, 5.5, "3. Proposed WiFi-CSI Sensing", border_color=C_PRIMARY)
    tb3 = s2.shapes.add_textbox(Inches(9.0), Inches(2.0), Inches(3.3), Inches(4.5))
    tf3 = tb3.text_frame
    tf3.word_wrap = True
    p = tf3.paragraphs[0]
    p.text = "Core Advantages:"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = C_PRIMARY
    bullets3 = [
        "100% Privacy-Preserving: Captures zero images or voice recordings.",
        "Through-Wall Capability: 2.4 GHz RF waves penetrate drywall and brick.",
        "Device-Free & Contactless: User wears nothing and needs zero interaction.",
        "Ultra Low Cost: Built with two ESP32-S3 boards (~INR 3,000 total).",
        "Multi-Purpose: Turns existing ambient Wi-Fi infrastructure into an active sensor."
    ]
    for b in bullets3:
        bp = tf3.add_paragraph()
        bp.text = f"• {b}"
        bp.font.size = Pt(11)
        bp.font.color.rgb = C_TEXT_DARK

    # =========================================================================
    # SLIDE 3: RF PHYSICS & CSI FUNDAMENTALS (Speaker 1)
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "WiFi-CSI Fundamentals: Physical Principle & Multipath Propagation", "RF Physical Layer", "Speaker 1")

    # Formula Card Top
    add_card(s3, 0.8, 1.4, 11.7, 2.0, "Frequency-Domain Channel Estimation in OFDM")
    tb = s3.shapes.add_textbox(Inches(1.0), Inches(1.9), Inches(11.3), Inches(1.4))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Channel Estimation Model:   Y_k = H_k · X_k + N_k   ==>   Ĥ_k = Y_k / X_k"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = C_PRIMARY
    p_desc = tf.add_paragraph()
    p_desc.text = "Wi-Fi (802.11n) transmits over orthogonal subcarriers. The receiver estimates the complex transmission coefficient H_k = |H_k| · e^(jφ_k) from the known Long Training Field (LTF) symbols in every packet preamble."
    p_desc.font.size = Pt(12)
    p_desc.font.color.rgb = C_TEXT_DARK

    # Card Left: Multipath Fading
    add_card(s3, 0.8, 3.6, 5.7, 3.4, "Multipath Decomposition & Human Motion")
    tb = s3.shapes.add_textbox(Inches(1.0), Inches(4.1), Inches(5.3), Inches(2.7))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Channel Sum:   H_k(t) = H_k^s (static room) + H_k^d(t) (human)"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = C_SECONDARY
    pts = [
        "Static Paths (H^s): Reflections off stationary walls, floor, and furniture.",
        "Dynamic Path (H^d): Bounces off the moving human body.",
        "Constructive/Destructive Interference: Amplitude oscillates between |H^s| - |H^d| and |H^s| + |H^d|.",
        "Walking creates periodic cyclic swings; Falling causes an abrupt transient spike followed by a new static state."
    ]
    for pt in pts:
        bp = tf.add_paragraph()
        bp.text = f"• {pt}"
        bp.font.size = Pt(11)

    # Card Right: Wavelength & Fresnel Zones
    add_card(s3, 6.8, 3.6, 5.7, 3.4, "Wavelength & Doppler Sensitivity")
    tb = s3.shapes.add_textbox(Inches(7.0), Inches(4.1), Inches(5.3), Inches(2.7))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "At 2.462 GHz:   λ = c / f = 3×10^8 / 2.462×10^9 ≈ 12.2 cm"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = C_PRIMARY
    pts2 = [
        "Phase Shift: Moving the body by λ/4 ≈ 3 cm changes the reflected round-trip path by 6 cm (λ/2 = 180° phase flip).",
        "Sub-Centimeter Sensitivity: Even subtle breathing or sitting posture changes produce detectable phase shifts.",
        "Doppler Shift: f_D = 2v / λ. Walking at 1 m/s produces a ~16 Hz Doppler fluctuation, well within our 100 Hz sampling rate.",
        "CSI vs. RSSI: RSSI is 1 scalar average; CSI provides 90 separate frequency measurements, resolving multipath."
    ]
    for pt in pts2:
        bp = tf.add_paragraph()
        bp.text = f"• {pt}"
        bp.font.size = Pt(11)

    # =========================================================================
    # SLIDE 4: HARDWARE SETUP & TESTBED (Speaker 1)
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Experimental Hardware Architecture & Through-Wall Testbed", "System Implementation", "Speaker 1")

    # Card 1: Transmitter
    add_card(s4, 0.8, 1.4, 3.6, 5.5, "Transmitter (TX)")
    tb = s4.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(3.2), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True
    tx_items = [
        ("Hardware", "ESP32-S3 Development Board (Xtensa LX7 @ 240 MHz)."),
        ("Antenna", "6 dBi omnidirectional 2.4 GHz dipole antenna."),
        ("Protocol", "ESP-NOW broadcast (connectionless, 0 association latency)."),
        ("Rate", "100 packet send attempts/second (T = 10 ms period)."),
        ("PHY Mode", "802.11n HT40 (40 MHz bandwidth) on Channel 11 (2.462 GHz)."),
        ("Fixed MAC", "1A:00:00:00:00:00 (locally administered address for receiver filtering).")
    ]
    for k, v in tx_items:
        p = tf.add_paragraph()
        p.text = f"{k}: {v}"
        p.font.size = Pt(11)

    # Card 2: Physical Testbed Layout
    add_card(s4, 4.8, 1.4, 3.6, 5.5, "Through-Wall Testbed Setup", border_color=C_PRIMARY)
    tb = s4.shapes.add_textbox(Inches(5.0), Inches(2.0), Inches(3.2), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True
    testbed_items = [
        ("Partition Obstruction", "Dense visual partition board blocking optical LOS."),
        ("Geometry", "TX on one side, person in center, RX on opposite side (~3–4 m distance)."),
        ("Activities", "Empty room, Sitting, Standing, Walking, Falling."),
        ("Safety Grounding", "Gym crash mats for human subject safety during falling trials."),
        ("Why 40 MHz (HT40)?", "Doubles bandwidth from 20 to 40 MHz, capturing 128 HT-LTF subcarriers for finer multipath resolution.")
    ]
    for k, v in testbed_items:
        p = tf.add_paragraph()
        p.text = f"{k}: {v}"
        p.font.size = Pt(11)

    # Card 3: Receiver & Host
    add_card(s4, 8.8, 1.4, 3.7, 5.5, "Receiver (RX) & Host PC")
    tb = s4.shapes.add_textbox(Inches(9.0), Inches(2.0), Inches(3.3), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True
    rx_items = [
        ("Hardware", "ESP32-S3 board in promiscuous Wi-Fi station mode."),
        ("Directional RX", "8 dBi ALFA directional patch antenna (concentrates sensitivity toward sensing area)."),
        ("CSI Extraction", "ESP-IDF `esp_wifi_set_csi_rx_cb()` hardware callback."),
        ("Payload", "384 bytes (192 complex I/Q subcarrier pairs per packet)."),
        ("Serial Transport", "921,600 baud USB-serial to Host PC."),
        ("Offline Processing", "Python signal processing pipeline and PyTorch deep neural networks.")
    ]
    for k, v in rx_items:
        p = tf.add_paragraph()
        p.text = f"{k}: {v}"
        p.font.size = Pt(11)

    # =========================================================================
    # SLIDE 5: CSI PREPROCESSING & STANDARDIZATION (Speaker 2)
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "CSI Data Engineering: From Raw Bytes to Model-Ready Tensors", "Data Engineering", "Speaker 2")

    # 4 Steps Cards in a Row
    steps = [
        ("1. Validation & Parsing", "Keep valid CSI_DATA lines with exactly 25 comma-separated fields and 384 signed 8-bit integers.", "Rejects corrupt packets; ensures integrity."),
        ("2. Amplitude Extraction", "Pair raw integers into in-phase (I) and quadrature (Q): |H_k| = sqrt(I^2 + Q^2).", "Amplitude is immune to random CFO/SFO oscillator phase drift."),
        ("3. Null-Carrier Drop", "Drop zero amplitudes corresponding to DC subcarriers and guard bands (leaves ~108 valid carriers).", "Eliminates dead hardware zones."),
        ("4. Resampling to 90", "Piecewise linear interpolation maps active subcarriers onto 90 standardized positions.", "Guarantees fixed tensor dimension across all packets.")
    ]
    for idx, (st, desc, why) in enumerate(steps):
        add_card(s5, 0.8 + idx * 2.95, 1.4, 2.8, 3.2, st)
        tb = s5.shapes.add_textbox(Inches(0.9 + idx * 2.95), Inches(2.0), Inches(2.6), Inches(2.4))
        tf = tb.text_frame
        tf.word_wrap = True
        p1 = tf.paragraphs[0]
        p1.text = desc
        p1.font.size = Pt(11)
        p1.font.color.rgb = C_TEXT_DARK
        p2 = tf.add_paragraph()
        p2.text = f"\nWhy: {why}"
        p2.font.size = Pt(10)
        p2.font.bold = True
        p2.font.color.rgb = C_PRIMARY

    # Bottom Banner: Temporal Windowing
    add_card(s5, 0.8, 4.8, 11.7, 2.2, "Temporal Windowing & Model Input Shape: [Batch, 1, 250, 90]", border_color=C_PRIMARY)
    tb = s5.shapes.add_textbox(Inches(1.0), Inches(5.3), Inches(11.3), Inches(1.5))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Window Length W = 250 packets (~2.5 seconds @ 100 Hz)   |   Stride S = 50 packets (0.5 second advance, 80% overlap)"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = C_SECONDARY
    p_sub = tf.add_paragraph()
    p_sub.text = "• 2.5 seconds of context captures complete sit-to-stand transitions, several walking strides, or a complete fall-and-rest sequence.\n• Overlap multiplies training samples and prevents critical events from being severed at window boundaries.\n• Final model-ready tensor per sample: 250 time packets × 90 amplitude features."
    p_sub.font.size = Pt(11)

    # =========================================================================
    # SLIDE 6: DATA AUDIT & LEAKAGE DISCOVERY (Speaker 2)
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Critical Scientific Audit: Discovering Window-Level Data Leakage", "Reproducibility Audit", "Speaker 2")

    # Left: The Discovery Card
    add_card(s6, 0.8, 1.4, 5.7, 5.5, "The Overlap Leakage Diagnosis", border_color=C_ALERT)
    tb = s6.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.3), Inches(4.7))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Critical Audit Finding:"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = C_ALERT
    leak_pts = [
        "The legacy dataset had 2,241 windows created with stride 50 (80% overlap = 200 packets shared between adjacent windows).",
        "The 70/15/15 train/val/test split was performed AT THE WINDOW LEVEL.",
        "Consequently, Window i in Train shared 80% of identical raw packets with Window i+1 in Test!",
        "Over 90% of test windows had near-identical sister windows in the training set.",
        "The model easily achieved 97%–98% test accuracy by interpolating adjacent frames and memorizing the room's static multipath fingerprint."
    ]
    for pt in leak_pts:
        bp = tf.add_paragraph()
        bp.text = f"• {pt}"
        bp.font.size = Pt(11)

    # Right: The Empirical Proof Card
    add_card(s6, 6.8, 1.4, 5.7, 5.5, "Empirical Proof: Generalization Collapse")
    tb = s6.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.3), Inches(4.7))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Testing 97.63% Checkpoint on 25 Unseen Recordings:"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = C_SECONDARY
    proof_pts = [
        "We collected 25 new unseen recording sessions in `live_recordings/`.",
        "When evaluated on these unseen recordings, accuracy collapsed from 97.63% to 20.55% (no better than random guessing across 5 classes!).",
        "Recording-level majority vote accuracy: only 20.00% (5 of 25 correct).",
        "Why? Receiver Automatic Gain Control (AGC) shifted amplitude ranges (max jumped from 25.88 to 277.19), saturating uncalibrated inputs.",
        "Conclusion: Prior 98% accuracy claims measured memorization, not true generalized through-wall sensing."
    ]
    for pt in proof_pts:
        bp = tf.add_paragraph()
        bp.text = f"• {pt}"
        bp.font.size = Pt(11)

    # =========================================================================
    # SLIDE 7: DATASET ENRICHMENT & ZERO-LEAKAGE PROTOCOL (Speaker 2)
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Dataset Enrichment & The Leakage-Resistant Grouped Protocol", "Data Engineering", "Speaker 2")

    # Table Card Top
    add_card(s7, 0.8, 1.4, 11.7, 3.0, "Dataset Expansion: 51 Original Recordings + 25 New Recordings = 76 Sessions")
    tb = s7.shapes.add_textbox(Inches(1.0), Inches(1.9), Inches(11.3), Inches(2.3))
    tf = tb.text_frame
    tf.word_wrap = True
    dist_text = """Class       | Original Files | Original Windows | New Files | New Windows | Total Files | Total Windows | Class Share
Empty       | 20             | 841              | 5         | 118         | 25          | 959           | 34.4%
Sitting     | 10             | 471              | 5         | 108         | 15          | 579           | 20.7%
Standing    | 9              | 409              | 5         | 92          | 14          | 501           | 18.0%
Walking     | 11             | 477              | 5         | 125         | 16          | 602           | 21.6%
Falling (!) | 1              | 43               | 5         | 107         | 6           | 150           | 5.4%
Total       | 51             | 2,241            | 25        | 550         | 76          | 2,791         | 100.0%"""
    p = tf.paragraphs[0]
    p.text = dist_text
    p.font.name = "Consolas"
    p.font.size = Pt(10)

    # Card Left: Leakage-Free Protocol
    add_card(s7, 0.8, 4.6, 5.7, 2.4, "The Recording-Grouped Protocol", border_color=C_PRIMARY)
    tb = s7.shapes.add_textbox(Inches(1.0), Inches(5.1), Inches(5.3), Inches(1.8))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "• Recording-Level Quarantine: Entire CSV files are assigned to Train, Val, or Test BEFORE windowing.\n• Strictly Zero Overlap: 0% packet sharing between train and test sets.\n• Train: 53 files (1,945 windows) | Val: 11 files (369 windows) | Test: 12 files (477 windows)."
    p.font.size = Pt(11)

    # Card Right: Train-Fitted Normalization
    add_card(s7, 6.8, 4.6, 5.7, 2.4, "Train-Only Normalization & Class Weights")
    tb = s7.shapes.add_textbox(Inches(7.0), Inches(5.1), Inches(5.3), Inches(1.8))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "• Scaling statistics (min, max, μ, σ) are computed EXCLUSIVELY on training recordings and frozen for test and live inference.\n• Falling Class Weighting: w_c = N / (C · N_c) assigns a 3.78× loss penalty to missed falls, preventing the model from ignoring the minority class."
    p.font.size = Pt(11)

    # =========================================================================
    # SLIDE 8: MODEL ARCHITECTURE EVOLUTION (Speaker 3)
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Model Evolution: Grounding Neural Networks in RF Physics", "Deep Learning Architectures", "Speaker 3")

    # Left: The Input Layout Correction
    add_card(s8, 0.8, 1.4, 5.7, 5.5, "1. The Crucial Input Layout Correction")
    tb = s8.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.3), Inches(4.7))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "The Wrong Layout [B, 250, 90]:"
    p.font.bold = True
    p.font.size = Pt(12)
    p.font.color.rgb = C_ALERT
    wrong_desc = tf.add_paragraph()
    wrong_desc.text = "• PyTorch Conv1D treats dimension 1 as channels.\n• [B, 250, 90] treated 250 time packets as unordered channels and slid filters across the 90 subcarriers.\n• Temporal order was destroyed in the first layer! Accuracy capped at 91.10%."
    wrong_desc.font.size = Pt(11)

    p2 = tf.add_paragraph()
    p2.text = "\nThe Correct Layout [B, 90, 250]:"
    p2.font.bold = True
    p2.font.size = Pt(12)
    p2.font.color.rgb = C_PRIMARY
    right_desc = tf.add_paragraph()
    right_desc.text = "• Permute tensor to [B, 90, 250].\n• 90 CSI subcarriers are channels; Conv1D slides along 250 time steps.\n• Acts as a learned temporal filter bank detecting Doppler oscillations across frequency.\n• Single change increased accuracy from 91.10% to 97.63%!"
    right_desc.font.size = Pt(11)

    # Right: Baseline CNN-BiLSTM Architecture
    add_card(s8, 6.8, 1.4, 5.7, 5.5, "2. Original Baseline: CNN-BiLSTM (319k Params)")
    tb = s8.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.3), Inches(4.7))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Layer-by-Layer Architecture:"
    p.font.bold = True
    p.font.size = Pt(12)
    arch_pts = [
        "Input: [B, 1, 250, 90] -> squeeze + permute -> [B, 90, 250]",
        "Conv1D 1: 90 -> 64 channels (kernel 5, padding 2) + BN + ReLU (28,864 params)",
        "Conv1D 2: 64 -> 128 channels (kernel 3, padding 1) + BN + ReLU (24,704 params)",
        "MaxPool1D: kernel 2, stride 2 (halves time: 250 -> 125 steps)",
        "BiLSTM: input 128, hidden 128, 1 layer (2 directions: 264,192 params)",
        "Concat States: ht[-2] (forward) + ht[-1] (backward) = 256 features",
        "Classifier: Dropout(0.1) + Linear(256 -> 5) (1,285 params)",
        "Total Parameters: 319,429 | Historical Acc: 96.54% ± 2.03%"
    ]
    for pt in arch_pts:
        bp = tf.add_paragraph()
        bp.text = f"• {pt}"
        bp.font.size = Pt(11)

    # =========================================================================
    # SLIDE 9: PROPOSED NOVEL ARCHITECTURES (Speaker 3)
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Proposed Novel Architectures: MS-CNN-BiGRU & SE-TCN", "Architectural Innovation", "Speaker 3")

    # Card Left: MS-CNN-BiGRU
    add_card(s9, 0.8, 1.4, 5.7, 5.5, "Proposed 1: MS-CNN-BiGRU (188k Params)", border_color=C_PRIMARY)
    tb = s9.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.3), Inches(4.7))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Multi-Scale Temporal Filter Bank:"
    p.font.bold = True
    p.font.size = Pt(12)
    p.font.color.rgb = C_PRIMARY
    msg_pts = [
        "Multi-Scale Convolutions: Parallel branches with kernels k=3 (30 ms fall spikes), k=5 (walking cadence), and k=7 (slow posture shifts).",
        "Subcarrier SE Attention: Squeeze-and-Excitation dynamically suppresses low-SNR fading notch subcarriers.",
        "Bidirectional GRU (hidden 96): Eliminates separate cell state, saving 33% recurrent parameters.",
        "Parameter Reduction: 188,673 params (41% smaller than 319k baseline).",
        "Legacy Test Accuracy: 97.09% ± 0.95% (cuts baseline variance in half!)."
    ]
    for pt in msg_pts:
        bp = tf.add_paragraph()
        bp.text = f"• {pt}"
        bp.font.size = Pt(11)

    # Card Right: SE-TCN Champion
    add_card(s9, 6.8, 1.4, 5.7, 5.5, "Proposed 2: SE-TCN Champion (110k Params)", border_color=C_ACCENT)
    tb = s9.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.3), Inches(4.7))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Non-Recurrent Dilated Temporal Network:"
    p.font.bold = True
    p.font.size = Pt(12)
    p.font.color.rgb = C_ACCENT
    tcn_pts = [
        "Fully Convolutional: No recurrent bottlenecks; fully parallelized on CPU with 2.75 ms latency.",
        "Exponential Dilations (d = 1, 2, 4, 8): Receptive field spans the entire 2.5s window without vanishing gradients.",
        "Global Average Pooling: Eliminates dense linear parameters and prevents temporal boundary overfitting.",
        "Extreme Efficiency: Only 110,165 parameters (65.5% smaller than baseline).",
        "Unseen Session Accuracy: 58.57% ± 2.64% (+9.18% higher than CNN-BiLSTM on true generalization!)."
    ]
    for pt in tcn_pts:
        bp = tf.add_paragraph()
        bp.text = f"• {pt}"
        bp.font.size = Pt(11)

    # =========================================================================
    # SLIDE 10: EXPERIMENTAL RESULTS & ABLATIONS (Speaker 4)
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Experimental Results: Controlled Multi-Seed Benchmarks", "Empirical Evaluation", "Speaker 4")

    # Master Table Card Top
    add_card(s10, 0.8, 1.4, 11.7, 3.2, "Multi-Seed Benchmark: Legacy Split vs. Leakage-Free Grouped Split")
    tb = s10.shapes.add_textbox(Inches(1.0), Inches(1.9), Inches(11.3), Inches(2.5))
    tf = tb.text_frame
    tf.word_wrap = True
    res_text = """Model               | Parameters | Latency | Legacy Random Split Acc | Grouped Leakage-Free Acc | Grouped Bal. Acc | Grouped Fall Recall
MLP                 | 11,588,101 | 1.37 ms | 89.91%                  | 46.33%                   | 39.67%           | 26.32%
LeNet (2D CNN)      | 163,813    | 4.00 ms | 96.44%                  | 52.41%                   | 42.23%           | 0.00% (Blind to falls)
BiLSTM              | 80,517     | 1.56 ms | 72.11%                  | 41.93%                   | 48.67%           | 94.74% (Severe false alarms)
CNN_GRU             | 153,669    | 2.76 ms | 97.63%                  | 45.28%                   | 35.39%           | 0.00% (Blind to falls)
CNN_BiLSTM (Base)   | 319,429    | 2.39 ms | 94.36% ± 2.16%          | 49.39% ± 3.89%           | 46.26%           | 45.26%
MS_CNN_BiGRU (Ours) | 188,673    | 12.8 ms | 97.09% ± 0.95%          | 50.02% ± 4.63%           | 45.28%           | 29.47%
SE_TCN (Champion)   | 110,165    | 2.75 ms | 97.15% ± 0.61%          | 58.57% ± 2.64% (Best 62%)| 55.46%           | 47.37%"""
    p = tf.paragraphs[0]
    p.text = res_text
    p.font.name = "Consolas"
    p.font.size = Pt(9.5)

    # Card Left: Ablation Loss
    add_card(s10, 0.8, 4.8, 5.7, 2.2, "Ablation A: Loss Function for Fall Sensitivity", border_color=C_PRIMARY)
    tb = s10.shapes.add_textbox(Inches(1.0), Inches(5.3), Inches(5.3), Inches(1.6))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "• Standard Cross-Entropy: Fall Recall = 10.53% (missed 17 of 19 falls!).\n• Class-Weighted CE: Fall Recall = 15.79% (Macro-F1 improves to 0.4874).\n• Focal Loss (γ = 2.0): Fall Recall = 31.58% (3× higher fall sensitivity!)."
    p.font.size = Pt(11)

    # Card Right: Ablation Normalization
    add_card(s10, 6.8, 4.8, 5.7, 2.2, "Ablation B: Normalization Scheme Stability", border_color=C_SECONDARY)
    tb = s10.shapes.add_textbox(Inches(7.0), Inches(5.3), Inches(5.3), Inches(1.6))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "• Train Min-Max: 58.07% accuracy across unseen recordings.\n• Z-Score Standardization ((x - μ)/σ): 58.49% accuracy (best gain stability).\n• Dynamic Demean (x - x_mean): 49.27% (confirms static baseline carries sitting/standing height profile)."
    p.font.size = Pt(11)

    # =========================================================================
    # SLIDE 11: LITERATURE COMPARISON & STROHMAYER 2023 (Speaker 4)
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Literature Comparison: Strohmayer & Kampel (2023) vs. Our System", "Literature Context", "Speaker 4")

    # Card Left: Base Paper
    add_card(s11, 0.8, 1.4, 5.7, 5.5, "Base Paper: Strohmayer & Kampel (ICVS 2023)")
    tb = s11.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.3), Inches(4.7))
    tf = tb.text_frame
    tf.word_wrap = True
    base_pts = [
        ("Hardware", "2× ESP32 with hand-built copper biquad antennas."),
        ("Radio", "ESP-NOW broadcast @ 100 Hz, 20 MHz bandwidth (52 L-LTF subcarriers)."),
        ("Environment", "18 m hallway through 4 structural brick walls."),
        ("Task", "3 coarse classes: Empty, Walking, Walk+Wave."),
        ("Model", "EfficientNetV2-S (2D Image CNN with ~20 Million parameters)."),
        ("Reported Result", "97.43% ± 1.1% (LOS) / 88.16% ± 4.6% (Through 4 walls)."),
        ("Key Limitation", "Heavy 20M parameter model; only 3 coarse dynamic movements; no fall detection.")
    ]
    for k, v in base_pts:
        p = tf.add_paragraph()
        p.text = f"{k}: {v}"
        p.font.size = Pt(11)

    # Card Right: Our Work
    add_card(s11, 6.8, 1.4, 5.7, 5.5, "Our Work: ESP32-S3 + SE-TCN / MS-CNN-BiGRU", border_color=C_PRIMARY)
    tb = s11.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.3), Inches(4.7))
    tf = tb.text_frame
    tf.word_wrap = True
    our_pts = [
        ("Hardware", "2× ESP32-S3 boards with 6 dBi omni TX and 8 dBi patch RX."),
        ("Radio", "ESP-NOW broadcast @ 100 Hz, 40 MHz HT40 (192 subcarrier entries -> 90 features)."),
        ("Environment", "Short range through opaque visual partition board."),
        ("Task", "5 fine-grained classes: Empty, Sitting, Standing, Walking, Falling."),
        ("Model", "SE-TCN (110k params) & MS-CNN-BiGRU (188k params) — 180× more compact!"),
        ("Results", "97.15% ± 0.61% (Legacy split) / 58.57% ± 2.64% (Unseen sessions)."),
        ("Defense Point", "We do not claim our model outperforms Strohmayer; our setup evaluates a harder 5-class problem at 180× smaller model size.")
    ]
    for k, v in our_pts:
        p = tf.add_paragraph()
        p.text = f"{k}: {v}"
        p.font.size = Pt(11)

    # =========================================================================
    # SLIDE 12: LIVE DEMO & CONCLUSION (Speaker 4)
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "Live Inference Pipeline, Practical Limitations & Future Roadmap", "Deployment & Viva Defense", "Speaker 4")

    # Card Left: Live Pipeline
    add_card(s12, 0.8, 1.4, 5.7, 5.5, "Live Streaming Inference Engine", border_color=C_PRIMARY)
    tb = s12.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.3), Inches(4.7))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Operational Architecture:"
    p.font.bold = True
    p.font.size = Pt(12)
    p.font.color.rgb = C_PRIMARY
    live_pts = [
        "FIFO Buffer: Maintains rolling queue of the latest 250 packets (~2.5s).",
        "Pipeline Consistency: Applies exact amplitude extraction, null-drop, interpolation to 90 features, and training-fitted normalization.",
        "Inference Cadence: Evaluates every 50 packets (0.5s stride) with 2.75 ms compute latency.",
        "Prediction Smoothing: 3-window majority vote prevents visual flickering.",
        "Eldercare Fall Alert: Triggers emergency dispatch alert when P(Falling) >= 0.50 followed by prolonged stillness."
    ]
    for pt in live_pts:
        bp = tf.add_paragraph()
        bp.text = f"• {pt}"
        bp.font.size = Pt(11)

    # Card Right: Limitations & Conclusion
    add_card(s12, 6.8, 1.4, 5.7, 5.5, "Honest Limitations & Project Conclusion")
    tb = s12.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.3), Inches(4.7))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Limitations & Future Roadmap:"
    p.font.bold = True
    p.font.size = Pt(12)
    p.font.color.rgb = C_SECONDARY
    fut_pts = [
        "Cross-Environment Domain Shift: Tested in single room; moving furniture changes static multipath (addressed via few-shot fine-tuning).",
        "Multi-Person Interference: Superimposed reflections require multi-antenna Angle-of-Arrival (AoA) decomposition.",
        "Phase Sanitization: Adding sanitized phase differences between packets is listed as future work.",
        "Microcontroller Quantization: At 110k parameters, SE_TCN is ready for INT8 quantization on the ESP32-S3's 8 MB PSRAM.",
        "Conclusion: Validated that commodity ESP32-S3 Wi-Fi provides a robust, privacy-preserving, camera-free sensing modality."
    ]
    for pt in fut_pts:
        bp = tf.add_paragraph()
        bp.text = f"• {pt}"
        bp.font.size = Pt(11)

    # =========================================================================
    # SLIDE 13: THANK YOU & Q&A
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    bg13 = s13.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg13.fill.solid()
    bg13.fill.fore_color.rgb = C_DARK_BG
    bg13.line.color.rgb = C_DARK_BG

    tb = s13.shapes.add_textbox(Inches(2.0), Inches(2.2), Inches(9.3), Inches(3.0))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "THANK YOU!"
    p.font.size = Pt(44)
    p.font.bold = True
    p.font.color.rgb = RGBColor(255, 255, 255)
    p.alignment = PP_ALIGN.CENTER

    p2 = tf.add_paragraph()
    p2.text = "\nPrivacy-Preserving Through-Wall Human Activity Recognition Using WiFi-CSI"
    p2.font.size = Pt(18)
    p2.font.color.rgb = C_PRIMARY
    p2.alignment = PP_ALIGN.CENTER

    p3 = tf.add_paragraph()
    p3.text = "Group 10 • Dept. of ECE, RNS Institute of Technology • 2026-2027\nQuestions & Panel Discussion"
    p3.font.size = Pt(14)
    p3.font.color.rgb = C_TEXT_MUTED
    p3.alignment = PP_ALIGN.CENTER

    output_path = "WiFi_CSI_HAR_Final_Presentation.pptx"
    prs.save(output_path)
    print(f"Presentation saved successfully to: {output_path}")


if __name__ == "__main__":
    create_deck()
