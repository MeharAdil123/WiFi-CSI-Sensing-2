"""
Generates the comprehensive Viva Defense Handbook & Examination Q&A Guide:
WiFi_CSI_HAR_Viva_Questions_and_Answers.pdf

Built with ReportLab Platypus.
Covers 36+ Predictable, Technical, and Hostile Panel Questions with Model Answers,
RF Physics Proofs, Team Division Strategy for 4 Members, and Scientific Defense Rules.
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
            self.drawString(54, 750, "WiFi-CSI HAR Viva Examination & Defense Handbook — ECE Dept, RNSIT (2026-27)")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.75)
            self.line(54, 742, 558, 742)

            # Footer
            self.line(54, 45, 558, 45)
            self.drawString(54, 32, "Confidential — Viva Panel Defense Reference | Group 10")
            page_str = f"Page {self._pageNumber} of {page_count}"
            self.drawRightString(558, 32, page_str)

        self.restoreState()


def build_pdf():
    pdf_filename = "WiFi_CSI_HAR_Viva_Questions_and_Answers.pdf"
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom color palette
    c_primary = colors.HexColor("#0D9488")     # Teal accent
    c_navy = colors.HexColor("#0F172A")        # Deep navy
    c_slate = colors.HexColor("#334155")       # Dark text
    c_alert = colors.HexColor("#BE123C")       # Deep crimson for hostile questions
    c_green = colors.HexColor("#047857")       # Emerald for model answers
    c_bg_light = colors.HexColor("#F8FAFC")    # Very light gray
    c_bg_callout = colors.HexColor("#F1F5F9")  # Soft slate
    c_bg_hostile = colors.HexColor("#FFF1F2")  # Light red

    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=23,
        leading=28,
        textColor=c_navy,
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=c_primary,
        spaceAfter=14
    )
    h1_style = ParagraphStyle(
        "H1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=c_navy,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=c_primary,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=c_slate,
        spaceAfter=5
    )
    body_bold = ParagraphStyle(
        "BodyBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=13,
        textColor=c_navy,
        spaceAfter=3
    )
    q_predictable = ParagraphStyle(
        "QPredictable",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=c_navy,
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True
    )
    q_hostile = ParagraphStyle(
        "QHostile",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=c_alert,
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True
    )
    ans_style = ParagraphStyle(
        "AnsStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=c_slate,
        spaceAfter=5
    )
    key_takeaway = ParagraphStyle(
        "KeyTakeaway",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=12,
        textColor=c_green,
        spaceAfter=6
    )

    story = []

    # =========================================================================
    # COVER / HEADER BLOCK
    # =========================================================================
    story.append(Paragraph("WiFi-CSI Through-Wall HAR: Complete Viva Examination & Panel Defense Handbook", title_style))
    story.append(Paragraph("Master Guide of 36+ Predictable, Technical, and Hostile Panel Questions with Rigorous Scientific Answers", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=c_primary, spaceBefore=0, spaceAfter=10))

    meta_table_data = [
        [
            Paragraph("<b>Institution:</b> RNS Institute of Technology (RNSIT)", body_style),
            Paragraph("<b>Department:</b> Electronics & Communication Engineering", body_style)
        ],
        [
            Paragraph("<b>Project:</b> Privacy-Preserving Through-Wall HAR via WiFi-CSI", body_style),
            Paragraph("<b>Team:</b> 4 Members (Group 10) | Academic Year 2026-2027", body_style)
        ],
        [
            Paragraph("<b>Hardware:</b> Dual ESP32-S3 (2.4 GHz HT40, 100 Hz ESP-NOW)", body_style),
            Paragraph("<b>Architectures:</b> Baseline CNN-BiLSTM, MS-CNN-BiGRU, Champion SE-TCN", body_style)
        ]
    ]
    meta_table = Table(meta_table_data, colWidths=[250, 254])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_bg_light),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 0: TEAM STRATEGY & GOLDEN RULES OF VIVA DEFENSE
    # =========================================================================
    story.append(Paragraph("0. Viva Presentation Strategy & Golden Rules for 4 Team Members", h1_style))
    story.append(Paragraph(
        "A university viva panel tests two things: (1) <i>Did you build and understand this yourself?</i> and (2) <i>Can you defend your design decisions with scientific honesty?</i> "
        "Divide ownership clearly across your 4 team members so the panel sees total synchronization:",
        body_style
    ))

    team_data = [
        [Paragraph("<b>Member</b>", body_bold), Paragraph("<b>Primary Ownership & Role</b>", body_bold), Paragraph("<b>Target Questions & Hand-off</b>", body_bold)],
        [
            Paragraph("<b>Member 1</b><br/>(System & RF)", body_style),
            Paragraph("Problem statement, privacy motivation, RF wave propagation, multipath fading, CSI vs RSSI physics, ESP32-S3 hardware setup, ESP-NOW protocol.", body_style),
            Paragraph("Takes all RF physics, antenna, wall penetration, frequency (2.4 GHz), and baud rate questions.", body_style)
        ],
        [
            Paragraph("<b>Member 2</b><br/>(Data & Integrity)", body_style),
            Paragraph("CSI packet parsing, amplitude extraction, subcarrier interpolation to 90 features, windowing (250 steps), <b>Data Leakage diagnosis</b>, and Leakage-free Grouped Split.", body_style),
            Paragraph("Takes all questions on data leakage, session splits, sliding window overlap, and Z-score normalization.", body_style)
        ],
        [
            Paragraph("<b>Member 3</b><br/>(AI Architecture)", body_style),
            Paragraph("Model architectures: Baseline CNN-BiLSTM, Proposed MS-CNN-BiGRU, Champion SE-TCN, Dilated Causal Convolutions, and Squeeze-and-Excitation attention.", body_style),
            Paragraph("Takes all questions on neural network layers, receptive field calculation, parameter count (110k), and attention weights.", body_style)
        ],
        [
            Paragraph("<b>Member 4</b><br/>(Evaluation & Demo)", body_style),
            Paragraph("Empirical benchmarks (5 seeds), Focal Loss for Fall Detection, ablation studies, live real-time inference pipeline (2.75 ms latency), comparison with Strohmayer (2023).", body_style),
            Paragraph("Drives the live demonstration script (`live_demo.py`), explains confusion matrices, fall recall, and edge feasibility.", body_style)
        ]
    ]
    t_team = Table(team_data, colWidths=[80, 224, 200])
    t_team.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0D9488")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_team)
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "<b>Three Golden Rules of Viva Defense:</b><br/>"
        "1. <b>Never Claim Perfection:</b> If asked why test accuracy on unseen sessions is 58.57% rather than 98%, proudly answer: <i>'98% was an artifact of random window splitting data leakage. 58.57% is the true, scientifically verified cross-session generalization on completely unseen recordings, outperforming the baseline by +9.18%.'</i> Panels respect scientific honesty above fake 99% accuracy.<br/>"
        "2. <b>Smooth Hand-offs:</b> If an examiner asks Member 1 about dilated convolutions, Member 1 should briefly introduce it and pass to Member 3: <i>'Our architecture relies on dilated convolutions to expand receptive field; Member 3, who designed the SE-TCN block, can walk through the exact mathematical dilation schedule.'</i><br/>"
        "3. <b>Anchor in Physical Numbers:</b> Always quote exact parameters: 110,165 parameters, 2.75 ms latency, 250×90 input tensor, 100 Hz sampling rate, 64 subcarriers, 47.37% fall recall with Focal Loss.",
        body_style
    ))

    # =========================================================================
    # SECTION 1: FUNDAMENTAL RF & WIFI SENSING QUESTIONS (PREDICTABLE)
    # =========================================================================
    story.append(Spacer(1, 8))
    story.append(Paragraph("1. Category 1: Fundamental RF & WiFi Sensing Questions (Predictable)", h1_style))

    qa_cat1 = [
        (
            "Q1: What is the fundamental physical difference between CSI and RSSI? Why can't we just use RSSI for HAR?",
            "RSSI (Received Signal Strength Indicator) is a single scalar number (e.g., -55 dBm) measuring the total radio energy averaged across the entire channel bandwidth. It suffers from severe multipath superposition: when signals bounce off walls, ceiling, and body, they sum destructively or constructively into a single fluctuating number. RSSI cannot distinguish whether a signal drop was caused by a human standing up, walking away, or simple destructive interference.<br/>"
            "CSI (Channel State Information), by contrast, reports fine-grained PHY-layer subcarrier-level channel frequency response: <b>H_k = |H_k| exp(j ∠H_k)</b> for each orthogonal subcarrier <i>k</i>. Because different subcarrier wavelengths travel slightly different spatial paths, human movement creates unique, high-dimensional frequency-selective fading patterns across the 64 subcarriers. CSI provides a 'spatial-temporal spectrogram' of motion, enabling micro-activity classification.",
            "Key Defense Takeaway: RSSI is a 1-pixel scalar blur; CSI is a 64-frequency spatial-temporal spectrum of the environment."
        ),
        (
            "Q2: How does radio wave multipath propagation enable through-wall activity recognition?",
            "When the ESP32-S3 transmitter emits 2.4 GHz electromagnetic waves, the signal travels along hundreds of distinct paths: the direct Line-of-Sight (LoS) path through the partition wall, and multiple Non-Line-of-Sight (NLoS) reflected paths off walls, floor, furniture, and the human body.<br/>"
            "The human body consists of ~70% saline water, making it a strong dielectric reflector (dielectric constant ε_r ≈ 50). When a person moves (e.g., transitions from standing to sitting or falling), the length of the reflected radio paths changes continuously. This dynamic path length change causes phase shifts (Doppler effect) and constructive/destructive interference, altering the channel frequency response H(f, t) in a manner uniquely characteristic of each activity.",
            "Key Defense Takeaway: The human body acts as a dynamic dielectric mirror modulating the wireless multipath profile."
        ),
        (
            "Q3: Why did you choose 2.4 GHz rather than 5 GHz or 60 GHz mmWave?",
            "RF penetration through solid obstacles depends inversely on frequency. According to Maxwell's equations and Friis transmission physics, RF attenuation through construction materials (concrete, drywall, wood) increases substantially at higher frequencies.<br/>"
            "At 2.4 GHz (λ ≈ 12.2 cm), radio waves easily penetrate drywall, wooden partitions, and standard interior doors with low attenuation (~3–5 dB per wall). At 5 GHz (λ ≈ 6 cm), attenuation doubles (~8–12 dB), and at 60 GHz mmWave, signals cannot penetrate standard walls at all. Furthermore, 2.4 GHz hardware (ESP32-S3) is ubiquitous, low-power, and costs less than $5.",
            "Key Defense Takeaway: 2.4 GHz provides the optimal trade-off between through-wall penetration depth and Doppler motion sensitivity."
        ),
        (
            "Q4: What are Fresnel zones, and why are they relevant to WiFi activity recognition?",
            "Fresnel zones are concentric ellipsoidal regions of constructive and destructive interference surrounding the direct transmitter-receiver axis. The radius of the <i>n</i>-th Fresnel zone at distance d1 from TX and d2 from RX is: <b>r_n = √((n · λ · d1 · d2) / (d1 + d2))</b>.<br/>"
            "When a human enters the 1st Fresnel zone, the reflected path length differs from the LoS path by less than λ/2, causing strong constructive/destructive interference. By tracking how subcarrier amplitudes oscillate as the person crosses successive Fresnel boundaries, the neural network extracts the velocity, direction, and magnitude of body limb movement.",
            "Key Defense Takeaway: Crossing Fresnel boundaries translates physical body movement into rhythmic subcarrier amplitude waves."
        ),
        (
            "Q5: Why did you discard CSI phase and exclusively utilize CSI amplitude?",
            "Raw CSI phase <b>∠H_k = atan2(Q_k, I_k)</b> is heavily corrupted by two unsynchronized hardware artifacts on low-cost commercial microcontrollers: (1) <i>Carrier Frequency Offset (CFO)</i> caused by clock drift between TX and RX local oscillators, and (2) <i>Sampling Time Offset (STO)</i> caused by ADC timing jitter and packet detection delays. These offsets introduce random, time-varying linear phase slopes across subcarriers that completely mask human motion.<br/>"
            "While linear phase unwrapping algorithms (like sanitization via linear regression) exist, on the ESP32-S3 they frequently introduce residual phase noise. CSI amplitude <b>|H_k| = √(I_k² + Q_k²)</b> is mathematically invariant to CFO and STO, providing a stable, motion-correlated representation without requiring microsecond synchronization.",
            "Key Defense Takeaway: CSI amplitude is robust to oscillator clock drift; raw phase is unusable without complex calibration."
        )
    ]

    for q, ans, takeaway in qa_cat1:
        story.append(Paragraph(q, q_predictable))
        story.append(Paragraph(f"<b>Model Answer:</b> {ans}", ans_style))
        story.append(Paragraph(takeaway, key_takeaway))
        story.append(Spacer(1, 4))

    # =========================================================================
    # SECTION 2: HARDWARE SETUP & ESP32-S3 IMPLEMENTATION (TECHNICAL)
    # =========================================================================
    story.append(Spacer(1, 8))
    story.append(Paragraph("2. Category 2: Hardware Setup, ESP32-S3, & Data Acquisition (Technical)", h1_style))

    qa_cat2 = [
        (
            "Q6: Why did you select the ESP32-S3 instead of research-grade Intel 5300 NIC or USRP SDRs?",
            "Most early WiFi sensing papers (2013-2020) used Intel 5300 Network Interface Cards or Atheros AR9580 chips. However: (1) Intel 5300 requires ancient PCI-e slots, Linux kernel 2.6/3.x, and bulky desktop PCs consuming 200W; (2) USRP Software Defined Radios cost $2,000 to $5,000 each and require high-bandwidth USB3/Ethernet links. Neither is viable for real-world ambient assisted living.<br/>"
            "The ESP32-S3 is a dual-core Xtensa 32-bit LX7 microcontroller costing under $5, consuming < 1 Watt, with built-in 2.4 GHz 802.11b/g/n radio and hardware CSI capture API (`esp_wifi_set_csi_rx_cb`). Demonstrating robust through-wall sensing on $5 hardware proves commercial deployability.",
            "Key Defense Takeaway: Intel 5300 is an obsolete $500 lab artifact; ESP32-S3 proves real-world, low-cost commercial viability."
        ),
        (
            "Q7: Why use the ESP-NOW protocol instead of connecting through a standard WiFi router?",
            "Standard Wi-Fi connections (TCP/IP via an Access Point) suffer from variable transmission rates, CSMA/CA back-off delays, beacon frame congestion, channel switching, and network overhead. If other devices stream 4K video, the packet arrival interval becomes erratic.<br/>"
            "ESP-NOW is a connectionless, peer-to-peer MAC-layer protocol developed by Espressif. It sends raw vendor-specific action frames directly between TX and RX without router pairing or IP handshakes. This guarantees a deterministic, jitter-free transmission rate of exactly 100 packets per second (10 ms period) on a fixed channel (Channel 11, HT40).",
            "Key Defense Takeaway: ESP-NOW provides connectionless, deterministic 100 Hz transmission free from router traffic interference."
        ),
        (
            "Q8: What is the subcarrier structure of 802.11n HT40 on ESP32-S3, and why interpolate to 90 features?",
            "In 802.11n High Throughput 40 MHz mode (HT40), the channel bandwidth is 40 MHz with 128 subcarriers spaced 312.5 kHz apart. However, not all 128 subcarriers contain active data: guard band subcarriers at the channel edges and the DC central subcarrier carry zero energy (null subcarriers). Depending on frame type (HT-LTF vs legacy), the ESP32 reports between 64 and 108 valid complex subcarrier pairs (I, Q).<br/>"
            "Because packet header fluctuations can occasionally alter the number of non-zero active carriers, our preprocessing pipeline filters out null subcarriers and resamples the remaining active carriers onto a uniform 90-point grid using 1D linear interpolation. This guarantees a strictly uniform tensor shape of [250, 90] across every recording.",
            "Key Defense Takeaway: Resampling standardizes variable active subcarriers into a fixed, uniform 90-feature vector."
        ),
        (
            "Q9: What serial baud rate is used between ESP32-S3 and the host PC, and why?",
            "The receiver ESP32-S3 connects to the host PC via USB-UART running at <b>921,600 baud</b>. Standard 115,200 baud is completely inadequate: each CSI packet contains ~200 bytes of raw binary payload (header, RSSI, noise floor, 128 I/Q bytes). At 100 packets/sec, this generates ~20,000 bytes/sec (~160,000 bits/sec). At 115,200 baud, the serial buffer overflows, dropping >30% of packets. Running at 921,600 baud provides a safe 5× throughput margin.",
            "Key Defense Takeaway: High-speed 921,600 baud serial UART prevents buffer overflow and packet dropping at 100 Hz."
        )
    ]

    for q, ans, takeaway in qa_cat2:
        story.append(Paragraph(q, q_predictable))
        story.append(Paragraph(f"<b>Model Answer:</b> {ans}", ans_style))
        story.append(Paragraph(takeaway, key_takeaway))
        story.append(Spacer(1, 4))

    # =========================================================================
    # SECTION 3: PREPROCESSING, DATA LEAKAGE, & SCIENTIFIC INTEGRITY
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("3. Category 3: Preprocessing, Data Leakage, & Scientific Integrity (Critical)", h1_style))
    story.append(Paragraph(
        "<i>Examiners love probing data leakage because 90% of undergraduate projects accidentally leak data and report bogus 99% accuracy. This section is your strongest weapon.</i>",
        body_style
    ))

    qa_cat3 = [
        (
            "Q10: Why did you choose a temporal window of 250 packets with a stride of 50?",
            "Our ESP-NOW transmission rate is 100 packets per second. A window length of <b>250 packets corresponds exactly to 2.50 seconds</b> of physical time. Biomechanically, human everyday activities have distinct durations: sitting down or standing up takes 1.5 to 2.2 seconds; a sudden fall takes 0.6 to 1.2 seconds followed by impact and stillness; walking requires several gait cycles (step period ~0.6s). A 2.5-second window captures the entire temporal transition.<br/>"
            "A stride of 50 packets (0.5 seconds advance, 80% overlap) ensures smooth temporal continuity and provides 5 classification updates per second during live inference.",
            "Key Defense Takeaway: 250 packets = 2.5 seconds, capturing full biomechanical motion transitions with 5 Hz live updates."
        ),
        (
            "Q11: What was the critical 'Data Leakage' defect in the legacy pipeline, and how did you diagnose it?",
            "The legacy pipeline segmented continuous recording files into 250-packet windows with a 50-packet stride (80% overlap = 200 packets shared between consecutive windows). It then performed an unstructured <b>random shuffle split</b> (80% train, 20% test) across all windows.<br/>"
            "Because adjacent windows shared 200 identical raw CSI packets, window <i>i</i> fell in the training set while overlapping window <i>i+1</i> fell in the test set. Over <b>90% of test windows shared identical physical packets with training windows</b>. The neural network was not learning generalized human motion; it was memorizing specific packet sequences, yielding an artificially inflated accuracy of 97.63%!<br/>"
            "We proved this by evaluating the legacy checkpoint on 25 completely new, unseen recording sessions: accuracy collapsed catastrophically to <b>20.55%</b> (pure chance).",
            "Key Defense Takeaway: Random window splitting causes 80% packet sharing between train and test, masquerading memorization as accuracy."
        ),
        (
            "Q12: If test accuracy dropped from 97.63% to 58.57% on the grouped split, did your project fail? (Hostile)",
            "<i>(Deliver this answer with total confidence):</i><br/>"
            "No, absolutely not. The 97.63% figure was an illusion caused by data leakage in the previous implementation. A model reporting 97% that collapses to 20% on new data is a failed, useless model.<br/>"
            "Our rigorous <b>Leakage-Free Grouped Split</b> partitions entire <i>recording sessions</i> (76 sessions grouped by recording ID) strictly into Train (70%), Val (15%), and Test (15%). No packet from any test recording is ever seen during training. Achieving <b>58.57% accuracy and 55.46% balanced accuracy on completely unseen through-wall sessions</b> across 5 complex classes is an authentic, scientifically validated result. Furthermore, our proposed SE-TCN improved generalization by <b>+9.18%</b> over the baseline CNN-BiLSTM (49.39% -> 58.57%) and boosted Fall Recall from 10.5% to 47.4%. That is genuine scientific progress.",
            "Key Defense Takeaway: We replaced a leaked 97% illusion with an honest 58.6% cross-session generalization, proving real learning."
        ),
        (
            "Q13: Why is Z-Score Standardization superior to Min-Max Normalization across different sessions?",
            "Min-Max normalization rescales values using absolute global extrema: <b>x_norm = (x - min) / (max - min)</b>. However, commercial WiFi chips utilize Automatic Gain Control (AGC) hardware amplifiers that adjust receiver gain dynamically based on background RF noise. If a single burst of interference causes an amplitude spike (outlier), Min-Max squashes all genuine human motion signals into a microscopic dynamic range.<br/>"
            "Furthermore, if a new session has an amplitude value higher than `global_max`, Min-Max clips or exceeds [0, 1]. <b>Z-score standardization: z = (x - μ_train) / σ_train</b> relies on the distribution mean and standard deviation, making it robust to gain shifts and outliers. In our ablation experiments, Z-Score achieved 58.49% accuracy versus 56.12% for Min-Max.",
            "Key Defense Takeaway: Z-Score standardization is resilient to WiFi AGC amplifier jumps; Min-Max is destroyed by outliers."
        ),
        (
            "Q14: How did you ensure zero normalization data leakage during preprocessing?",
            "In many naive pipelines, `StandardScaler()` or `MinMaxScaler()` is fitted on the entire dataset before splitting, which leaks test distribution statistics (μ and σ) into training. In our pipeline (`src/dataset.py`), the normalizer is fitted <b>strictly on the training split only</b>. The computed training mean μ_train and standard deviation σ_train are then saved and applied as fixed constants to transform the validation and test splits.",
            "Key Defense Takeaway: Normalizer parameters are strictly computed on training data and frozen for test inference."
        )
    ]

    for q, ans, takeaway in qa_cat3:
        story.append(Paragraph(q, q_hostile if "fail" in q else q_predictable))
        story.append(Paragraph(f"<b>Model Answer:</b> {ans}", ans_style))
        story.append(Paragraph(takeaway, key_takeaway))
        story.append(Spacer(1, 4))

    # =========================================================================
    # SECTION 4: DEEP LEARNING ARCHITECTURES & MODEL DESIGN
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("4. Category 4: Deep Learning Architectures & Model Design", h1_style))

    qa_cat4 = [
        (
            "Q15: Walk through the Baseline CNN-BiLSTM architecture layer-by-layer.",
            "The baseline architecture processes input tensor [Batch, 250, 90]:<br/>"
            "1. <b>Conv1D Block 1:</b> 90 input channels -> 64 filters, kernel size 5, stride 1, padding 2, followed by BatchNorm1d and ReLU. Extracts local subcarrier frequency correlations.<br/>"
            "2. <b>Conv1D Block 2:</b> 64 -> 128 filters, kernel size 3, padding 1, BatchNorm1d, ReLU.<br/>"
            "3. <b>MaxPool1D:</b> Kernel 2, stride 2. Downsamples temporal length from 250 to 125 steps, reducing computational load.<br/>"
            "4. <b>Bidirectional LSTM:</b> 128 input features, 64 hidden units, 2 layers, dropout 0.15. The forward LSTM models forward temporal evolution; the backward LSTM models reverse dynamics. Concatenated output dimension = 128 (64 forward + 64 backward).<br/>"
            "5. <b>Classification Head:</b> Linear layer 128 -> 64 with ReLU and Dropout(0.2), followed by final Linear layer 64 -> 5 logits.",
            "Key Defense Takeaway: Baseline uses Conv1D for subcarrier spatial extraction and BiLSTM for temporal sequence modeling."
        ),
        (
            "Q16: What are the fundamental computational drawbacks of BiLSTM for edge deployment?",
            "LSTMs suffer from three major bottlenecks: (1) <i>Sequential Dependency:</i> Hidden state h_t cannot be computed until h_{t-1} is finished, preventing parallel GPU/DSP execution; (2) <i>Vanishing/Exploding Gradients:</i> Even with gating, backpropagation through 125 time steps over multiple layers degrades long-range gradients; (3) <i>High Parameter & Memory Footprint:</i> BiLSTM requires 8 separate weight matrices per layer (4 forward, 4 backward), totaling ~210k parameters, leading to high RAM consumption on microcontrollers.",
            "Key Defense Takeaway: BiLSTM cannot be parallelized along time and incurs heavy memory overhead on resource-constrained hardware."
        ),
        (
            "Q17: Explain your proposed Champion Architecture: Squeeze-and-Excitation TCN (SE-TCN).",
            "SE-TCN replaces recurrent loops entirely with a purely convolutional feed-forward network consisting of:<br/>"
            "1. <b>Spatial Feature Projection:</b> Conv1D projecting 90 subcarriers to 64 channels.<br/>"
            "2. <b>Residual Dilated TCN Blocks:</b> 4 stacked residual blocks with exponentially increasing dilation factors: <b>d = [1, 2, 4, 8]</b>, kernel size k = 3. Each block contains: Dilated Conv1D -> BatchNorm -> ReLU -> Dropout -> Dilated Conv1D -> BatchNorm -> Residual Add.<br/>"
            "3. <b>Squeeze-and-Excitation (SE) Block:</b> Placed inside each residual block. Global Average Pooling squeezes time into a channel descriptor; two FC layers (reduction ratio r=8) generate dynamic channel attention weights in [0, 1] via Sigmoid activation.<br/>"
            "4. <b>Global Temporal Average Pooling & Head:</b> Averages the sequence across time into a 128-dim vector, followed by a linear projection to 5 classes.",
            "Key Defense Takeaway: SE-TCN combines dilated convolutions for massive receptive field with channel attention at only 110k parameters."
        ),
        (
            "Q18: What is the exact mathematical definition of Dilated Convolution, and why is it powerful?",
            "For a 1D sequence x[t] and filter w of size k with dilation factor d, the dilated convolution is defined as: <b>y[t] = ∑_{i=0}^{k-1} w[i] · x[t - d · i]</b>.<br/>"
            "Standard convolution has d=1 (adjacent taps). When d=2, the filter samples every second input; when d=8, it samples every eighth input. Stacking layers with dilations [1, 2, 4, 8] expands the network's Effective Receptive Field (ERF) <b>exponentially: ERF = 1 + ∑_{l=1}^L 2 · (k - 1) · d_l</b>. With k=3 and dilations [1, 2, 4, 8], our receptive field spans over 120 time steps (~1.2 seconds of motion) without losing temporal resolution or pooling away subtle fall signatures.",
            "Key Defense Takeaway: Dilated convolutions expand receptive field exponentially with depth while keeping parameter count tiny."
        ),
        (
            "Q19: What is the Squeeze-and-Excitation attention block doing in the physical context of WiFi CSI?",
            "In indoor multipath environments, different subcarriers experience vastly different signal-to-noise ratios (SNR): some subcarriers suffer deep destructive fading nulls (carrying pure noise), while others reflect strongly off the moving person (carrying rich Doppler motion).<br/>"
            "The SE block acts as an <b>adaptive subcarrier feature selector</b>: the Squeeze step aggregates global temporal energy across each channel (z_c = (1/T) ∑ x_c(t)), and the Excitation step passes this through an MLP to compute attention weights <b>s = σ(W2 · ReLU(W1 · z))</b>. The network dynamically scales up informative motion channels (s_c ≈ 1.0) and suppresses noisy fading nulls (s_c ≈ 0.0).",
            "Key Defense Takeaway: SE attention automatically amplifies motion-sensitive subcarrier channels and mutes fading nulls."
        ),
        (
            "Q20: Compare the computational complexity and parameter counts of all evaluated models.",
            "We systematically evaluated 6 architectures under identical training protocols:<br/>"
            "• <b>MLP Baseline:</b> 34,757 params | 42.1% acc | Cannot capture temporal dynamics.<br/>"
            "• <b>LeNet-1D:</b> 45,829 params | 46.8% acc | Receptive field too narrow.<br/>"
            "• <b>Baseline BiLSTM:</b> 174,469 params | 48.2% acc | High training latency, vanishing gradients.<br/>"
            "• <b>Baseline CNN-BiLSTM:</b> 210,053 params | 49.39% ± 3.89% acc | 5.82 ms latency.<br/>"
            "• <b>Proposed MS-CNN-BiGRU:</b> 188,293 params | 50.02% ± 4.63% acc | Multi-scale kernels (k=3,5,7), 10% fewer params than BiLSTM.<br/>"
            "• <b>Champion SE-TCN:</b> <b>110,165 params</b> | <b>58.57% ± 2.64% acc (Best: 61.84%)</b> | <b>2.75 ms latency</b>.<br/>"
            "SE-TCN delivers +9.18% higher generalization accuracy while requiring <b>48% fewer parameters</b> and half the inference latency of the baseline CNN-BiLSTM.",
            "Key Defense Takeaway: SE-TCN is nearly 2× smaller and 2× faster than CNN-BiLSTM while boosting test accuracy by +9.2%."
        )
    ]

    for q, ans, takeaway in qa_cat4:
        story.append(Paragraph(q, q_predictable))
        story.append(Paragraph(f"<b>Model Answer:</b> {ans}", ans_style))
        story.append(Paragraph(takeaway, key_takeaway))
        story.append(Spacer(1, 4))

    # =========================================================================
    # SECTION 5: LOSS FUNCTIONS, CLASS IMBALANCE, & FALL DETECTION
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("5. Category 5: Loss Functions, Class Imbalance, & Fall Detection", h1_style))

    qa_cat5 = [
        (
            "Q21: Why is Fall Detection the most critical yet most difficult class to recognize?",
            "In elderly ambient assisted living, falling is an emergency event with life-threatening consequences (fractures, hemorrhages). However, in any realistic dataset, falling represents severe <b>class imbalance</b>: in our legacy dataset, Falling had only 43 windows out of 2,241 (1.92% of the dataset)! In our expanded dataset, it represents 150 out of 2,791 windows (5.37%).<br/>"
            "Under standard Cross-Entropy Loss, the network minimizes overall loss by simply predicting majority classes (Empty, Walking), yielding 95% overall accuracy but an abysmal 10.53% Fall Recall (missing 9 out of 10 falls).",
            "Key Defense Takeaway: Class imbalance causes standard cross-entropy to ignore rare, life-critical fall events."
        ),
        (
            "Q22: How does Focal Loss work mathematically, and how did it resolve class imbalance?",
            "Focal Loss (Lin et al.) modifies standard cross-entropy: <b>FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)</b>.<br/>"
            "The focusing parameter <b>γ = 2.0</b> dynamically downweights well-classified 'easy' examples (like Empty, where p_t ≈ 0.99, modulating factor (1 - 0.99)² = 0.0001), driving their loss contribution to near zero. Conversely, for rare, hard fall samples where p_t is low, the modulating factor remains close to 1.0. This forces the optimizer to concentrate gradient updates on difficult fall patterns.<br/>"
            "In our controlled ablation experiment on unseen sessions, switching from Cross-Entropy to Focal Loss (γ=2.0) <b>tripled Fall Recall from 10.53% to 31.58%</b>, and with SE-TCN reached <b>47.37%</b>.",
            "Key Defense Takeaway: Focal Loss with γ=2.0 downweights easy empty rooms by 10,000×, tripling fall detection sensitivity."
        ),
        (
            "Q23: In elderly healthcare, what is worse: a False Positive or a False Negative? How does your system balance this?",
            "A <b>False Negative (missed fall) is fatal</b>: an elderly individual lies on the floor unconscious with no alert dispatched. A <b>False Positive (false alarm) is an inconvenience</b>: the caregiver receives a notification or checks a notification prompt.<br/>"
            "Therefore, in healthcare sensing, Recall (Sensitivity) on Falling must be prioritized over Precision. Our live inference engine (`src/live_inference.py`) implements a dual-stage safety mechanism: whenever the model outputs a Falling probability above an adjustable threshold (default 0.35 rather than 0.50), it triggers an immediate persistent Fall Alert buffer requiring 3 consecutive normal confirmations before resetting.",
            "Key Defense Takeaway: In healthcare, missed falls are catastrophic; our system tunes the decision threshold to prioritize high recall."
        )
    ]

    for q, ans, takeaway in qa_cat5:
        story.append(Paragraph(q, q_predictable))
        story.append(Paragraph(f"<b>Model Answer:</b> {ans}", ans_style))
        story.append(Paragraph(takeaway, key_takeaway))
        story.append(Spacer(1, 4))

    # =========================================================================
    # SECTION 6: HOSTILE & UNPREDICTABLE PANEL QUESTIONS (CURVEBALLS)
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("6. Category 6: Hostile, Skeptical, & Unpredictable Panel Curveballs", h1_style))
    story.append(Paragraph(
        "<i>These are aggressive questions designed to throw you off balance. Memorize the exact counters below.</i>",
        body_style
    ))

    qa_cat6 = [
        (
            "Q24: 'Why not just install a $15 smart camera or a PIR motion sensor? Isn't WiFi sensing overcomplicated?'",
            "<i>(Counter with privacy, through-wall, and lighting limitations):</i><br/>"
            "1. <b>Privacy Violation:</b> Cameras cannot be installed in private spaces where falls most frequently occur: bathrooms (wet tiles, high fall risk) and bedrooms. Elderly individuals reject cameras due to psychological discomfort. WiFi sensing is 100% privacy-preserving; it captures only abstract electromagnetic reflections, not visual images.<br/>"
            "2. <b>Line-of-Sight & Lighting:</b> Cameras fail in complete darkness, smoke, or when occluded by bed sheets or furniture. WiFi penetrates blankets, darkness, and non-metallic partitions.<br/>"
            "3. <b>PIR Inadequacy:</b> PIR sensors detect broad thermal motion; they cannot distinguish between walking, sitting, standing, or a fall, and cannot detect a person lying motionless after a fall.",
            "Key Defense Takeaway: Cameras violate bathroom privacy and fail in darkness; PIR sensors cannot classify activities or detect stationary falls."
        ),
        (
            "Q25: 'Was your deep learning model converted to NumPy? Why or why not?' (Direct prompt question)",
            "<i>(Clarify the fundamental technical distinction):</i><br/>"
            "No, the neural network model itself is <b>NOT converted to NumPy</b>. Deep learning models execute forward inference using optimized tensor computation engines: in our system, PyTorch tensors running on CPU/GPU via C++ LibTorch backends. NumPy is a general-purpose CPU array library that lacks automatic differentiation, CUDA tensor acceleration, and quantized edge graph optimization.<br/>"
            "However, <b>NumPy (.npy format) IS used extensively in our data storage pipeline</b>: raw serial CSI packets are parsed, filtered, interpolated into [250, 90] matrices, and cached as NumPy arrays (`x_train.npy`, `y_train.npy`) for ultra-fast, zero-copy binary disk loading during batch training.<br/>"
            "If edge deployment onto the ESP32-S3 microcontroller is required, the PyTorch model would be exported to <b>ONNX</b> or <b>TensorFlow Lite for Microcontrollers (TFLite-Micro)</b> using 8-bit integer quantization (INT8) or Espressif ESP-DL C-libraries, not converted to NumPy.",
            "Key Defense Takeaway: The model runs on PyTorch tensors; NumPy is used for fast data caching. Edge deployment uses TFLite-Micro or ESP-DL."
        ),
        (
            "Q26: 'Why didn't you use a Vision Transformer (ViT) or Attention Transformer like modern LLMs?'",
            "Transformers have quadratic computational complexity <b>O(T²)</b> with respect to sequence length and lack inductive bias. They require massive datasets (millions of samples) to avoid severe overfitting. On small-to-moderate temporal RF sensing datasets (~2,800 windows), Transformers overfit severely and memorize noise.<br/>"
            "Our Champion SE-TCN provides the exact benefit of attention (via Squeeze-and-Excitation channel gating) combined with the strong inductive bias of 1D convolutions (translation equivariance) with strictly linear complexity <b>O(T)</b>. SE-TCN trains stably in seconds and has only 110k parameters, whereas a Transformer would require >5 million parameters and overfit.",
            "Key Defense Takeaway: Transformers overfit on small RF datasets; TCNs offer linear complexity and strong spatial-temporal inductive bias."
        ),
        (
            "Q27: 'What happens if a pet (dog or cat) enters the room? Will it trigger a false alarm?'",
            "RF reflections depend on the physical radar cross-section (RCS) and water volume of the moving object. A domestic pet (5–15 kg) has a fraction of the dielectric volume of an adult human (60–80 kg).<br/>"
            "Furthermore, pet movement occurs at floor level (below 0.5 m), perturbing only the lower outer Fresnel zones with higher Doppler frequencies (rapid quadruped leg movement). In contrast, human torso transitions (standing to sitting or falling) involve high-mass, large-area displacement across the primary central Fresnel zone. While a large dog may produce minor amplitude fluctuations, its spectral signature is distinct from human torso falls.",
            "Key Defense Takeaway: Pets have much smaller dielectric radar cross-sections and distinct high-frequency quadruped Doppler signatures."
        ),
        (
            "Q28: 'How does your work compare to Julian Strohmayer & Martin Kampel's 2023 ICVS paper? Did you beat them?'",
            "<i>(Deliver a nuanced, scientifically defensible comparison):</i><br/>"
            "Julian Strohmayer and Martin Kampel (ICVS 2023) conducted an outstanding pioneering benchmark: they transmitted 100 Hz ESP-NOW packets through <b>4 structural brick walls across an 18-meter distance</b>, achieving 88.16% accuracy on <b>3 broad classes</b> (Empty, Walking, Walk+Wave) using <b>EfficientNetV2-S</b> (20.3 million parameters).<br/>"
            "We do NOT claim to have 'beaten' them on their setup, because our physical testbed used a single partition wall at 3–4 meters, and our recognition task is significantly harder: <b>5 fine-grained classes</b>, including differentiating Sitting from Standing and detecting sudden Falling.<br/>"
            "Our genuine architectural breakthrough over Strohmayer is <b>extreme computational efficiency</b>: Strohmayer's EfficientNetV2-S has <b>20,300,000 parameters</b> and requires an external server. Our SE-TCN achieves through-wall 5-class recognition with only <b>110,165 parameters (over 180× smaller!)</b> with a 2.75 ms latency, making direct on-device edge execution realistic.",
            "Key Defense Takeaway: Strohmayer sensed 3 classes through 4 walls with 20M params; we sense 5 classes with 110k params (180× more compact)."
        ),
        (
            "Q29: 'What happens if furniture is moved or humidity/weather changes?'",
            "Static furniture rearrangement alters the static multipath profile (DC offset), but does NOT generate dynamic Doppler frequency shifts. Because our model learns temporal variations (rate of change over 2.5 seconds) and applies Z-score standardization, static multipath shifts are largely subtracted out.<br/>"
            "However, long-term environmental drift (seasonal humidity affecting drywall dielectric permittivity) is a known challenge across all RF sensing literature. The solution, which we recommend in our future work, is <i>domain-adversarial neural networks (DANN)</i> or <i>temporal self-supervised pre-training</i> on unlabelled ambient WiFi traffic.",
            "Key Defense Takeaway: Dynamic motion features filter out static furniture changes; environmental drift can be addressed via domain adaptation."
        ),
        (
            "Q30: 'Can your system track multiple people performing different activities simultaneously?'",
            "In its current single-link configuration (1 TX and 1 RX), simultaneous multi-person activity recognition is fundamentally underdetermined: the receiver observes a single superposition of reflections: <b>H_total = H_person1 + H_person2 + H_static</b>. Decomposing this into separate activity labels requires Blind Source Separation (BSS).<br/>"
            "Multi-person tracking requires a <i>multi-link mesh topology</i> (e.g., 3 TX and 3 RX ESP32 nodes) to provide spatial diversity and independent spatial observation angles. For single-occupant elderly monitoring (e.g., studio apartment or assisted living room), our single-link setup is optimal, minimal, and unobtrusive.",
            "Key Defense Takeaway: Single-link systems target single-occupant privacy monitoring; multi-person tracking requires multi-node mesh diversity."
        )
    ]

    for q, ans, takeaway in qa_cat6:
        story.append(Paragraph(q, q_hostile))
        story.append(Paragraph(f"<b>Model Answer:</b> {ans}", ans_style))
        story.append(Paragraph(takeaway, key_takeaway))
        story.append(Spacer(1, 4))

    # =========================================================================
    # SECTION 7: LIVE DEMO & RAPID-FIRE FORMULA CHEAT SHEET
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("7. Live Demonstration Protocol & Rapid-Fire Math Reference", h1_style))

    story.append(Paragraph("<b>How to Run and Explain the Live Demonstration in Front of the Panel:</b>", h2_style))
    story.append(Paragraph(
        "To demonstrate live inference without requiring active ESP32 hardware plugged in during the oral presentation, use our playback mode:<br/>"
        "<code>python live_demo.py --mode playback --file recordings/Falling_01.csv --model checkpoints/se_tcn_grouped.pth --arch se_tcn</code><br/>"
        "Explain to the panel: <i>'This script creates a real-time FIFO ring buffer of 250 packets. As each 100 Hz packet arrives, it extracts amplitude, interpolates to 90 features, normalizes using train statistics, and runs SE-TCN inference every 0.5s. When the fall event occurs at packet 120, notice the instant alert triggered with 2.75 ms latency.'</i><br/>"
        "If hardware is plugged in, run: <code>python live_demo.py --mode serial --port COM4 --baud 921600</code>.",
        body_style
    ))
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Essential Mathematical Formulas Every Member Must Know:</b>", h2_style))

    math_data = [
        [Paragraph("<b>Concept</b>", body_bold), Paragraph("<b>Mathematical Formula</b>", body_bold), Paragraph("<b>Physical Meaning</b>", body_bold)],
        [
            Paragraph("CSI Channel Response", body_style),
            Paragraph("<b>H_k = |H_k| · exp(j ∠H_k)</b><br/>|H_k| = √(I_k² + Q_k²)", body_style),
            Paragraph("Reports attenuation and phase rotation imparted by wireless multipath onto subcarrier k.", body_style)
        ],
        [
            Paragraph("Fresnel Zone Radius", body_style),
            Paragraph("<b>r_n = √((n · λ · d1 · d2) / (d1 + d2))</b>", body_style),
            Paragraph("Boundaries of constructive/destructive multipath interference. Crossing boundaries generates amplitude pulses.", body_style)
        ],
        [
            Paragraph("Dilated Convolution", body_style),
            Paragraph("<b>y[t] = ∑_{i=0}^{k-1} w[i] · x[t - d · i]</b>", body_style),
            Paragraph("Expands receptive field exponentially with dilation d without increasing parameters or downsampling.", body_style)
        ],
        [
            Paragraph("SE Channel Attention", body_style),
            Paragraph("<b>s = σ(W_2 · ReLU(W_1 · z))</b><br/>z_c = (1/T) ∑ x_c(t)", body_style),
            Paragraph("Learns dynamic attention weights in [0, 1] to amplify motion-sensitive subcarriers and suppress fading nulls.", body_style)
        ],
        [
            Paragraph("Focal Loss", body_style),
            Paragraph("<b>FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)</b>", body_style),
            Paragraph("Downweights easy background samples (Empty) by factor (1 - p_t)^2 to force learning of rare falls.", body_style)
        ],
        [
            Paragraph("Balanced Accuracy", body_style),
            Paragraph("<b>BA = (1/C) ∑_{c=1}^C (TP_c / N_c)</b>", body_style),
            Paragraph("Unweighted mean of class recalls. Immune to dataset imbalance, unlike raw accuracy.", body_style)
        ]
    ]
    t_math = Table(math_data, colWidths=[100, 180, 224])
    t_math.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0D9488")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_math)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Viva Questions & Answers PDF saved successfully to: {pdf_filename}")


if __name__ == "__main__":
    build_pdf()
