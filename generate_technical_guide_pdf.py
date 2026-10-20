"""
Generates the comprehensive master technical guide:
WiFi_CSI_HAR_Master_Technical_Guide.pdf

Built with ReportLab Platypus.
Includes complete theoretical depth, RF physics, mathematical equations,
layer-by-layer architectural walkthroughs, and a full glossary with simple examples.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and display total page count."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#64748B"))

        # Skip headers/footers on cover page
        if self._pageNumber > 1:
            # Header
            self.drawString(54, 750, "WiFi-CSI HAR Master Technical Guide — ECE Dept, RNSIT (2026-27)")
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.75)
            self.line(54, 742, 558, 742)

            # Footer
            self.line(54, 45, 558, 45)
            self.drawString(54, 32, "Confidential — Pair Programming & Viva Technical Defense Reference")
            page_str = f"Page {self._pageNumber} of {page_count}"
            self.drawRightString(558, 32, page_str)

        self.restoreState()


def build_pdf():
    pdf_filename = "WiFi_CSI_HAR_Master_Technical_Guide.pdf"
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    c_primary = colors.HexColor("#0D9488")
    c_navy = colors.HexColor("#0F172A")
    c_slate = colors.HexColor("#334155")
    c_light = colors.HexColor("#F8FAFC")
    c_alert = colors.HexColor("#E11D48")

    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=30,
        textColor=c_navy,
        spaceAfter=8
    )
    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=c_primary,
        spaceAfter=15
    )
    h1_style = ParagraphStyle(
        "H1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        textColor=c_navy,
        spaceBefore=16,
        spaceAfter=8,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=c_primary,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=c_slate,
        spaceAfter=6
    )
    body_bold = ParagraphStyle(
        "BodyBold",
        parent=body_style,
        fontName="Helvetica-Bold"
    )
    code_style = ParagraphStyle(
        "CodeStyle",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=4
    )
    callout_style = ParagraphStyle(
        "Callout",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1E293B")
    )

    story = []

    # =========================================================================
    # COVER / TITLE BLOCK
    # =========================================================================
    story.append(Paragraph("RNS INSTITUTE OF TECHNOLOGY, BENGALURU", subtitle_style))
    story.append(Paragraph("Department of Electronics and Communication Engineering", subtitle_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Privacy-Preserving Through-Wall Human Activity Recognition Using WiFi-CSI", title_style))
    story.append(Paragraph("Master Technical Guide & Comprehensive Theoretical Reference for Examination Panel", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_primary, spaceAfter=12))

    meta_text = """
    <b>Academic Year:</b> 2026–2027 | <b>Group:</b> 10<br/>
    <b>Project Team:</b> Md Mehar Adil D H (1RN23EC094), Mohith Gowda M (1RN23EC095), Nihar Gowda (1RN23EC106), Sujal L G (1RN23EC169)<br/>
    <b>Project Guide:</b> Dr. Leena Chandrashekar, Assoc. Prof. | <b>Project Reviewer:</b> Dr. Srikanth G N, Assoc. Prof.<br/>
    <b>Hardware Setup:</b> 2× ESP32-S3 (Dual-core Xtensa LX7 @ 240 MHz, 2.4 GHz HT40, 100 Hz ESP-NOW)
    """
    story.append(Paragraph(meta_text, body_style))
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 1: EXECUTIVE SUMMARY & PHYSICAL FOUNDATIONS
    # =========================================================================
    story.append(Paragraph("1. Executive Summary & Physical Foundations", h1_style))
    story.append(Paragraph(
        "This project presents an end-to-end, low-cost through-wall Human Activity Recognition (HAR) and emergency "
        "fall detection system using Channel State Information (CSI) extracted from standard Wi-Fi packets transmitted "
        "between two ESP32-S3 microcontrollers. Unlike optical cameras that compromise user privacy or wearable devices "
        "that suffer from non-compliance, Wi-Fi sensing operates contactlessly, penetrates structural drywall and partitions, "
        "and captures zero visual data.",
        body_style
    ))
    story.append(Paragraph(
        "<b>The Multipath Channel Physics:</b> In an indoor environment, transmitted RF waves propagate along multiple paths "
        "(direct path, wall reflections, body reflections) before arriving at the receiver antenna. The complex baseband channel "
        "response on subcarrier <i>k</i> at time <i>t</i> is the superposition of <i>L</i> paths:",
        body_style
    ))
    story.append(Paragraph(
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>H_k(t) = ∑ a_i(t) · exp(-j [2π f_k τ_i(t) + φ_i(t)]) = H_k^s + H_k^d(t)</b>",
        code_style
    ))
    story.append(Paragraph(
        "where <i>H_k^s</i> represents the static channel vector (stationary walls, desks, floor) and <i>H_k^d(t)</i> represents "
        "the dynamic multipath component scattered by the human body. As a person moves, the path length <i>d_i(t)</i> changes. "
        "At 2.462 GHz (Channel 11), the RF wavelength is <b>λ = c / f ≈ 12.2 cm</b>. A body displacement of just 3 cm alters the "
        "reflected round-trip path by 6 cm (λ/2), causing a 180° phase flip that turns constructive interference into destructive "
        "interference. The measured amplitude fluctuates between |H^s| - |H^d| and |H^s| + |H^d|, generating dynamic Doppler "
        "signatures (f_D = 2v / λ) characteristic of specific human motor actions.",
        body_style
    ))

    # =========================================================================
    # SECTION 2: HARDWARE & RADIO PIPELINE
    # =========================================================================
    story.append(Spacer(1, 6))
    story.append(Paragraph("2. ESP32-S3 Hardware & Signal Acquisition Pipeline", h1_style))
    story.append(Paragraph(
        "The system uses two commodity Espressif ESP32-S3 development boards with external antennas:",
        body_style
    ))

    hw_data = [
        ["Subsystem", "Specification", "Engineering Rationale"],
        ["Microcontroller", "ESP32-S3-N16R8 (Dual Xtensa LX7 @ 240 MHz)", "Low cost, official ESP-IDF CSI driver API support, 8 MB PSRAM."],
        ["Transmitter (TX)", "6 dBi omnidirectional 2.4 GHz dipole antenna", "Uniformly illuminates the indoor sensing area."],
        ["Receiver (RX)", "8 dBi ALFA directional patch antenna", "Concentrates sensitivity toward monitored room, rejecting rear multipath."],
        ["Radio Protocol", "ESP-NOW Broadcast (Vendor Action Frames)", "Connectionless, zero pairing/association overhead, steady 100 packets/s."],
        ["PHY Channel", "Channel 11 (2.462 GHz primary), HT40 (40 MHz)", "HT40 bandwidth doubles subcarriers to 128 HT-LTF for finer multipath."],
        ["Packet Rate", "100 packets/sec (T = 10 ms period)", "Nyquist limit is 50 Hz; captures walking Doppler (~16 Hz) smoothly."]
    ]
    t_hw = Table(hw_data, colWidths=[1.3*inch, 2.5*inch, 3.2*inch])
    t_hw.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_navy),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8.5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [c_light, colors.white]),
        ('GRID', (0,0), (-1,-1), 0.5, c_navy),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_hw)

    # =========================================================================
    # SECTION 3: SIGNAL PROCESSING & NORMALIZATION
    # =========================================================================
    story.append(Spacer(1, 8))
    story.append(Paragraph("3. Preprocessing & Tensor Standardization Pipeline", h1_style))
    story.append(Paragraph(
        "Every received packet produces 384 signed 8-bit integers representing 192 complex subcarrier entries. "
        "The preprocessing engine transforms raw serial lines into fixed (250 × 90) tensors through six deterministic steps:",
        body_style
    ))
    story.append(Paragraph(
        "<b>Step 1 — Packet Validation:</b> Verify exactly 25 comma-separated CSV fields starting with the CSI_DATA marker and "
        "containing 384 finite integers. Corrupted lines are safely discarded.<br/>"
        "<b>Step 2 — Amplitude Extraction:</b> Pair integers into in-phase (I) and quadrature (Q) components: "
        "<b>|H_k| = √(I_k² + Q_k²)</b>. Raw phase is discarded because unsynchronized oscillator Carrier Frequency Offset (CFO) "
        "and Sampling Frequency Offset (SFO) introduce severe packet-to-packet random slope drift.<br/>"
        "<b>Step 3 — Null Subcarrier Elimination:</b> Direct Current (DC) central subcarriers and guard band edge carriers carry "
        "zero power and report 0.0. Dropping these prevents artificial edge discontinuities.<br/>"
        "<b>Step 4 — Linear Interpolation to Exactly 90 Features:</b> Active subcarriers (~108 to 114) are resampled onto a 90-point "
        "uniform grid: x'_j = x_i + (x_{i+1} - x_i) · [(v_j - u_i) / (u_{i+1} - u_i)]. This guarantees fixed dimensions compatible "
        "with benchmark standards.<br/>"
        "<b>Step 5 — Train-Fitted Normalization:</b> Scale inputs using parameters fitted EXCLUSIVELY on training recordings: "
        "x̂ = (x - x_min^train) / (x_max^train - x_min^train). Never normalize test data using test set bounds.<br/>"
        "<b>Step 6 — Temporal Windowing:</b> Packets are cut into overlapping windows: Length W = 250 packets (2.5 s), "
        "Stride S = 50 packets (0.5 s step). Output sample dimension: <b>[250 × 90]</b>.",
        body_style
    ))

    # =========================================================================
    # SECTION 4: THE DATA AUDIT & LEAKAGE ASSESSMENT
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("4. Scientific Audit: The Window-Level Data Leakage Discovery", h1_style))
    story.append(Paragraph(
        "<b>The Leakage Mechanism:</b> In the legacy pipeline, 2,241 windows were generated with stride 50 (80% overlap = 200 packets "
        "shared between adjacent windows). The 70/15/15 train/val/test split was performed <b>at the window level</b>. "
        "Consequently, Window <i>i</i> in Train shared 80% identical packets with Window <i>i+1</i> in Test! "
        "Over 90% of test windows had near-identical sister windows in the training set.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Empirical Proof of Generalization Failure:</b> When evaluating the baseline checkpoint (reporting 97.63% test accuracy "
        "on the random split) against 25 new unseen recordings in `live_recordings/`, <b>accuracy collapsed to 20.55%</b> "
        "(random chance across 5 classes is 20.00%). This proved that prior 98% numbers reflected memorization of room multipath, "
        "not generalized activity recognition.",
        body_style
    ))
    story.append(Paragraph(
        "<b>The Leakage-Resistant Recording-Grouped Protocol:</b> We quarantined entire CSV recordings into disjoint Train, Val, "
        "and Test subsets BEFORE windowing (strictly 0% window overlap across sets). We enriched the dataset with 25 new recordings, "
        "giving <b>76 total recording sessions</b> and expanding Falling windows from 43 to 150.",
        body_style
    ))

    # =========================================================================
    # SECTION 5: MODEL ARCHITECTURES & MATHEMATICAL SPECIFICATIONS
    # =========================================================================
    story.append(Spacer(1, 8))
    story.append(Paragraph("5. Model Architecture Specifications & Mathematical Mechanics", h1_style))
    story.append(Paragraph(
        "<b>The Input Layout Correction:</b> In PyTorch, `nn.Conv1d` treats dimension 1 as channels. The original repository code "
        "reshaped input to [B, 250, 90], treating 250 time packets as unordered channels and convolving along frequency. "
        "Permuting to <b>[B, 90, 250]</b> treats 90 subcarriers as channels and convolves along time, acting as a learned temporal filter bank. "
        "This single physical correction increased test accuracy from 91.10% to 97.63%.",
        body_style
    ))

    m_data = [
        ["Model Architecture", "Params", "Latency", "Legacy Acc (Mean±SD)", "Grouped Acc (Mean±SD)", "Key Technical Trait"],
        ["Baseline MLP", "11.58M", "1.37 ms", "89.91%", "46.33%", "Fully connected; severe temporal overfitting."],
        ["Baseline LeNet", "163k", "4.00 ms", "96.44%", "52.41%", "2D image CNN; isotropic kernels."],
        ["Baseline BiLSTM", "80k", "1.56 ms", "72.11%", "41.93%", "Pure recurrent; high-frequency noise."],
        ["Baseline CNN_GRU", "153k", "2.76 ms", "97.63%", "45.28%", "1D CNN + unidirectional GRU."],
        ["Baseline CNN_BiLSTM", "319k", "2.39 ms", "94.36 ± 2.16%", "49.39 ± 3.89%", "Original reference; 128 hidden BiLSTM."],
        ["Proposed MS_CNN_BiGRU", "188k", "12.7 ms", "97.09 ± 0.95%", "50.02 ± 4.63%", "Kernels 3,5,7 + SE Attention + BiGRU."],
        ["Proposed SE_TCN (Champ)", "110k", "2.75 ms", "97.15 ± 0.61%", "58.57 ± 2.64%", "Dilated causal convolutions; +9.18% over base!"]
    ]
    t_m = Table(m_data, colWidths=[1.5*inch, 0.65*inch, 0.65*inch, 1.3*inch, 1.35*inch, 1.55*inch])
    t_m.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_navy),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [c_light, colors.white]),
        ('GRID', (0,0), (-1,-1), 0.5, c_navy),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('TOPPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_m)

    # =========================================================================
    # SECTION 6: LOSS FUNCTIONS, ABLATIONS & OPTIMIZATION
    # =========================================================================
    story.append(Spacer(1, 8))
    story.append(Paragraph("6. Loss Formulations, Imbalance Ablations & Optimization", h1_style))
    story.append(Paragraph(
        "<b>Cross-Entropy vs. Focal Loss:</b> Standard Cross-Entropy loss is: L_CE = -log(p_y). When classes are imbalanced "
        "(Falling is ~5% of data), the majority classes dominate the loss gradient, causing the model to achieve 58% overall accuracy "
        "while detecting only 10.5% of real falls. <b>Focal Loss</b> adds a dynamic focusing parameter γ=2.0: "
        "<b>L_Focal = -α_t (1 - p_t)^γ log(p_t)</b>. When a fall sample has low predicted probability (p_t << 1), the modulating factor "
        "(1 - p_t)^2 approaches 1.0, preserving full gradient pressure. Our ablations proved that <b>Focal Loss tripled Fall Recall from 10.5% to 31.6%</b>.",
        body_style
    ))
    story.append(Paragraph(
        "<b>The Adam Optimizer Mechanics:</b> Adam maintains running first moments (momentum, β1=0.9) and second moments "
        "(uncentered variance, β2=0.999). After bias correction, parameter update is: θ_{t+1} = θ_t - [α / (√(v̂_t) + ε)] · m̂_t. "
        "This assigns adaptive step sizes to each parameter, stabilizing training across convolutional and recurrent layers.",
        body_style
    ))

    # =========================================================================
    # SECTION 7: COMPARISON WITH STROHMAYER & KAMPEL (2023)
    # =========================================================================
    story.append(Spacer(1, 8))
    story.append(Paragraph("7. Scientific Comparison with Strohmayer & Kampel (ICVS 2023)", h1_style))
    story.append(Paragraph(
        "Julian Strohmayer and Martin Kampel (ICVS 2023) demonstrated through-wall activity recognition using 2 ESP32 boards "
        "transmitting 100 Hz ESP-NOW packets through <b>4 structural brick walls across an 18-meter distance</b>, achieving "
        "<b>88.16% ± 4.6%</b> accuracy on 3 coarse classes (Empty, Walking, Walk+Wave) using an <b>EfficientNetV2-S</b> model (20M parameters).<br/>"
        "<b>Why higher accuracy on our dataset does NOT prove our model is superior:</b><br/>"
        "1. Our setup uses a single visual partition board at 3–4 meters, not 4 solid brick walls at 18 meters.<br/>"
        "2. Our dataset evaluates a harder 5-class problem (including Sitting vs. Standing and Falling).<br/>"
        "3. Our genuine contribution over Strohmayer is <b>extreme edge efficiency</b>: our SE_TCN model uses only <b>110,165 parameters</b> "
        "(over <b>180× smaller</b> than EfficientNetV2-S), making it viable for direct microcontroller deployment.",
        body_style
    ))

    # =========================================================================
    # SECTION 8: MASTER GLOSSARY & WORKED EXAMPLES
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("8. Master Technical Glossary: Definitions & Worked Examples", h1_style))
    story.append(Paragraph(
        "Every single technical and theoretical term used in the presentation, repository, and paper, explained with simple real-world examples:",
        body_style
    ))

    glossary = [
        ("OFDM (Orthogonal Frequency Division Multiplexing)", "Splits a wide radio channel into many narrow, orthogonal subcarriers sent in parallel.", "Like a wide highway divided into 128 narrow lanes where cars do not interfere."),
        ("Subcarrier", "One individual frequency lane within the OFDM channel (spaced 312.5 kHz apart).", "Subcarrier 0 is at 2.452 GHz, subcarrier 1 at 2.4523125 GHz."),
        ("CSI (Channel State Information)", "A complex matrix reporting how the wireless channel attenuated and delayed each subcarrier.", "One packet reports 192 complex numbers: H_k = I_k + jQ_k."),
        ("In-Phase (I) & Quadrature (Q)", "The real (cosine-aligned) and imaginary (sine-aligned) components of the downconverted RF baseband signal.", "If I = 3 and Q = 4, the complex coefficient is 3 + j4."),
        ("CSI Amplitude", "The absolute signal strength on subcarrier k: |H_k| = √(I² + Q²).", "For I=3, Q=4: Amplitude = √(9 + 16) = 5.0."),
        ("CSI Phase", "The angle of the signal vector: φ = atan2(Q, I).", "atan2(4, 3) ≈ 53.1°. Discarded because oscillator drift adds random offsets."),
        ("Multipath Fading", "Signal copies arriving over different paths with different delays, adding constructively or destructively.", "Walking through the room causes peaks and nulls as waves reinforce or cancel."),
        ("RF Wavelength (λ)", "Physical distance spanned by one full radio wave cycle: λ = c / f.", "At 2.462 GHz: λ = (3×10^8) / (2.462×10^9) ≈ 12.2 cm."),
        ("RSSI", "Received Signal Strength Indicator: a single scalar measuring total received power.", "-55 dBm. Averages all subcarriers together, losing spatial multipath details."),
        ("ESP-NOW", "Espressif's connectionless Wi-Fi MAC action frame protocol.", "TX broadcasts packets directly to RX every 10 ms without router pairing."),
        ("HT40", "High Throughput 40 MHz channel bandwidth (IEEE 802.11n).", "Doubles bandwidth from 20 MHz, giving 128 HT-LTF subcarriers for finer resolution."),
        ("Null Subcarrier", "Guard band edge carriers and DC central carrier carrying zero energy.", "Reported amplitudes [0, 5, 6, 0, 7] -> zeros dropped to keep active [5, 6, 7]."),
        ("Resampling (Linear Interpolation)", "Mapping variable active subcarriers onto a fixed 90-point grid by joining dots with straight lines.", "Maps ~108 active carriers to exactly 90 features so tensor size is constant."),
        ("Min-Max Normalization", "Rescaling values into [0, 1] using training set bounds: (x - min) / (max - min).", "Range [1.0, 25.88]: value 13.44 becomes (13.44 - 1)/(24.88) = 0.50."),
        ("Temporal Window", "A fixed block of consecutive packets treated as one model input sample.", "250 consecutive packets ≈ 2.5 seconds of motion at 100 packets/s."),
        ("Stride & Overlap", "Stride is how far the window advances; overlap is packets shared with adjacent windows.", "Window=250, Stride=50 -> Overlap = 250 - 50 = 200 packets (80% shared)."),
        ("Data Leakage", "Information from the test set contaminating training, creating artificially inflated accuracy.", "Window 0-249 in train and window 50-299 in test share 200 identical packets."),
        ("Conv1D", "Slides a small filter along a 1D sequence computing a learned weighted sum.", "Filter [1, 0, -1] on [2, 4, 6] produces 1·2 + 0·4 + (-1)·6 = -4."),
        ("Receptive Field", "How many original input time steps contribute to a single deep activation.", "After Conv1 (k=5) + Conv2 (k=3) + Pool (2), each activation sees ~8 packets (~80 ms)."),
        ("BiLSTM", "Two LSTMs: one reads time forward (0->125), one backward (125->0).", "Forward summarizes motion ending; backward summarizes from rest back to start."),
        ("BiGRU", "Bidirectional Gated Recurrent Unit: uses 2 gates (reset/update) instead of 3 gates.", "Saves 33% recurrent parameters over LSTM with equal memory retention."),
        ("Dilated Convolution", "Convolution where filter taps have gaps (dilation factor d).", "Dilation d=2 on filter [a, b, c] samples steps t-2, t, t+2, expanding field exponentially."),
        ("SE Attention", "Squeeze-and-Excitation: learns to dynamically scale informative channels.", "Assigns weights in [0, 1] to boost high-SNR subcarriers and suppress fading nulls."),
        ("Dropout", "Randomly zeros a fraction p of activations during training (disabled at test time).", "p=0.15: 15% of features set to zero, preventing neuron co-adaptation."),
        ("Logits", "Raw unconstrained real-valued scores output by the final linear layer.", "Logits [-1.2, 0.5, 3.4] are converted by softmax into probabilities."),
        ("Softmax", "Converts logits z into posterior probabilities that sum to 1.0: exp(z_c) / ∑exp(z).", "Logits [0, 2] -> exp(0)=1, exp(2)=7.39 -> Probabilities = [0.12, 0.88]."),
        ("Cross-Entropy Loss", "Negative log-likelihood of the true class: -log(p_y).", "If true class prob = 0.90, loss = -ln(0.90) = 0.105; if prob = 0.10, loss = 2.30."),
        ("Focal Loss", "Downweights easy examples with factor (1 - p_t)^γ to focus on hard minority classes.", "Tripled Fall Recall from 10.5% to 31.6% by forcing network to learn rare falls."),
        ("Adam Optimizer", "Maintains moving averages of gradient (momentum) and squared gradient (variance).", "Gives each parameter an adaptive learning rate; ideal for hybrid CNN-RNN models."),
        ("Early Stopping", "Halting training when validation metric stops improving for patience epochs.", "Patience 6: stops if validation Macro-F1 does not improve for 6 epochs, saving best model."),
        ("Balanced Accuracy", "Unweighted average of recall across all classes: (1/C) ∑ Recall_c.", "Unlike raw accuracy, balanced accuracy is unaffected by majority Empty samples."),
        ("Macro F1-Score", "Harmonic mean of precision and recall averaged equally across all 5 classes.", "A model with 99% accuracy on Empty but 0% on Falling will have poor Macro-F1 (~0.70)."),
        ("Confusion Matrix", "Table where rows are true classes and columns are predicted classes.", "Diagonal entries are correct classifications; off-diagonals are mistakes.")
    ]

    for term, definition, example in glossary:
        story.append(Paragraph(f"<b>{term}</b>", h2_style))
        story.append(Paragraph(f"<b>Meaning:</b> {definition}", body_style))
        story.append(Paragraph(f"<b>Real-World Example:</b> <i>{example}</i>", callout_style))
        story.append(Spacer(1, 3))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Master Technical Guide saved successfully to: {pdf_filename}")


if __name__ == "__main__":
    build_pdf()
