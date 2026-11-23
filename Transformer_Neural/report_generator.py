"""
PDF Report Generator for TCS Asset Inspection System.
Generates professional multi-page reports with TCS branding, asset images,
defect tables, labeled parts, and an overall summary.
"""

import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, PageBreak, HRFlowable,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

from config import TCS_LOGO_PATH, REPORT_DIR, InspectionResult


class ReportGenerator:
    """Generates professional PDF inspection reports with TCS branding."""

    # ── TCS Brand Colours ─────────────────────────────────────────────────
    NAVY    = colors.HexColor('#0A2240')
    GREEN   = colors.HexColor('#27ae60')
    RED     = colors.HexColor('#c0392b')
    ORANGE  = colors.HexColor('#e67e22')
    YELLOW  = colors.HexColor('#f39c12')
    GREY    = colors.HexColor('#666666')
    LIGHT   = colors.HexColor('#f0f4f8')
    DIVIDER = colors.HexColor('#cccccc')

    def __init__(self):
        os.makedirs(REPORT_DIR, exist_ok=True)
        print(f"[REPORT] Report generator initialized. Output: {REPORT_DIR}/")

    # ── Public API ────────────────────────────────────────────────────────

    def generate(self, results: list) -> str:
        """Generate a PDF report from a batch of InspectionResults."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"TCS_Inspection_Report_{timestamp}.pdf"
        filepath = os.path.join(REPORT_DIR, filename)

        doc = SimpleDocTemplate(
            filepath, pagesize=A4,
            rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40,
        )

        styles = getSampleStyleSheet()
        story = []

        # Cover page
        story.extend(self._cover_page(styles, results))
        story.append(PageBreak())

        # Per-asset pages
        for i, result in enumerate(results):
            story.extend(self._asset_page(styles, result, i + 1))
            if i < len(results) - 1:
                story.append(PageBreak())

        # Summary page
        story.append(PageBreak())
        story.extend(self._summary_page(styles, results))

        doc.build(story)
        print(f"\n[REPORT] [OK] Report generated: {filepath}")
        return filepath

    # ── Cover Page ────────────────────────────────────────────────────────

    def _cover_page(self, styles, results):
        elems = []

        # TCS logo at top of first page
        if os.path.exists(TCS_LOGO_PATH):
            logo = RLImage(TCS_LOGO_PATH, width=3.2 * inch, height=1.0 * inch)
            logo.hAlign = 'CENTER'
            elems.append(logo)
        else:
            hdr = ParagraphStyle('TCSHdr', parent=styles['Title'],
                                  fontSize=26, textColor=self.NAVY, spaceAfter=10)
            elems.append(Paragraph("TATA CONSULTANCY SERVICES", hdr))

        elems.append(Spacer(1, 25))
        elems.append(HRFlowable(width='85%', thickness=2.5, color=self.NAVY, spaceAfter=20))

        title = ParagraphStyle('Title2', parent=styles['Title'],
                                fontSize=24, textColor=colors.HexColor('#1a1a2e'),
                                spaceAfter=10, alignment=TA_CENTER)
        elems.append(Paragraph("Substation Asset Inspection Report", title))

        sub = ParagraphStyle('Sub', parent=styles['Normal'],
                              fontSize=13, textColor=self.GREY,
                              alignment=TA_CENTER, spaceAfter=25)
        elems.append(Paragraph("Autonomous Robot Vision Inspection System", sub))
        elems.append(Spacer(1, 20))

        now = datetime.now()
        faulty = sum(1 for r in results if r.overall_condition == 'Faulty')
        meta = [
            ['Report Date:', now.strftime('%d %B %Y')],
            ['Report Time:', now.strftime('%H:%M:%S IST')],
            ['Assets Inspected in Batch:', str(len(results))],
            ['Faulty Assets Detected:', str(faulty)],
            ['Healthy Assets:', str(len(results) - faulty)],
            ['Vision AI Engine:', 'YOLOv8 Real-Time Neural Detector'],
            ['Inspection Trigger Mode:', 'Asset ID Plate Recognition'],
            ['Target Parameters:', 'Oil Leakage, Cracks, Cuts, Burns, Loose Conns'],
        ]
        t = Table(meta, colWidths=[2.6 * inch, 3.2 * inch])
        t.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#333333')),
            ('TEXTCOLOR', (1, 0), (1, -1), self.NAVY),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
            ('TOPPADDING', (0, 0), (-1, -1), 7),
            ('LINEBELOW', (0, 0), (-1, -2), 0.5, colors.HexColor('#e0e0e0')),
        ]))
        t.hAlign = 'CENTER'
        elems.append(t)

        elems.append(Spacer(1, 40))
        note = ParagraphStyle('Note', parent=styles['Normal'],
                               fontSize=9, textColor=colors.HexColor('#888888'),
                               alignment=TA_CENTER)
        elems.append(Paragraph(
            'CONFIDENTIAL \u2014 Generated by TCS Autonomous Robot Vision Inspection System. '
            'Automated trigger after 4 assets analyzed.', note))

        return elems

    # ── Per-Asset Page ────────────────────────────────────────────────────

    def _asset_page(self, styles, result: InspectionResult, index: int):
        elems = []

        cond_color = self.RED if result.overall_condition == 'Faulty' else self.GREEN
        hdr = ParagraphStyle('AHdr', parent=styles['Heading1'],
                              fontSize=18, textColor=self.NAVY, spaceAfter=5)
        elems.append(Paragraph(
            f'Asset #{index}: {result.asset_id} \u2014 {result.asset_type}', hdr))

        badge = ParagraphStyle('Badge', parent=styles['Normal'],
                                fontSize=13, textColor=cond_color, spaceAfter=5)
        if result.overall_condition == 'Faulty':
            icon = 'FAULTY \u2014 DEFECTS DETECTED'
        else:
            icon = 'HEALTHY \u2014 NO DEFECTS DETECTED'
        elems.append(Paragraph(f'<b>Status: {icon}</b>', badge))

        ts = ParagraphStyle('TS', parent=styles['Normal'],
                             fontSize=9.5, textColor=self.GREY, spaceAfter=10)
        elems.append(Paragraph(
            f'Inspected: {result.timestamp}  |  YOLO Latency: {result.inference_time_ms:.1f}ms', ts))
        elems.append(HRFlowable(width='100%', thickness=1, color=self.DIVIDER, spaceAfter=10))

        # ── Images side-by-side ───────────────────────────────────────────
        iw, ih = 2.8 * inch, 2.2 * inch
        imgs = []
        if os.path.exists(result.image_path):
            imgs.append(RLImage(result.image_path, width=iw, height=ih))
        if result.annotated_image_path and os.path.exists(result.annotated_image_path):
            imgs.append(RLImage(result.annotated_image_path, width=iw, height=ih))

        if len(imgs) == 2:
            it = Table([imgs], colWidths=[3 * inch, 3 * inch])
            it.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            elems.append(it)
            lbl = ParagraphStyle('IL', parent=styles['Normal'],
                                  fontSize=9, alignment=TA_CENTER, textColor=self.GREY)
            elems.append(Table(
                [[Paragraph('Original Robot Capture', lbl), Paragraph('YOLO Labeled & Annotated Parts', lbl)]],
                colWidths=[3 * inch, 3 * inch]))
        elif imgs:
            elems.append(imgs[0])

        elems.append(Spacer(1, 12))

        # Oil gauge reading for transformers
        if result.oil_gauge_reading is not None:
            gc = self.GREEN if result.oil_gauge_reading >= 70 else self.RED
            gs = ParagraphStyle('Gauge', parent=styles['Normal'], fontSize=11, textColor=gc)
            diag = "Normal level" if result.oil_gauge_reading >= 70 else "Low oil level warning! Leakage detected."
            elems.append(Paragraph(
                f'<b>Oil Level Gauge Reading: {result.oil_gauge_reading}%</b> ({diag})', gs))
            elems.append(Spacer(1, 8))

        # Labeled parts / components identified
        if result.detected_components:
            comp_names = [c["class"].replace("tag_", "ID Plate: ").replace("_", " ").title() for c in result.detected_components]
            comp_str = ", ".join(set(comp_names))
            cs = ParagraphStyle('Comps', parent=styles['Normal'], fontSize=9.5, textColor=colors.HexColor('#444444'))
            elems.append(Paragraph(f'<b>Verified Components Labeled:</b> {comp_str}', cs))
            elems.append(Spacer(1, 8))

        # ── Defect table ──────────────────────────────────────────────────
        if result.findings:
            elems.append(Paragraph('<b>Detected Anomalies & Defects:</b>', styles['Heading3']))
            elems.append(Spacer(1, 4))

            rows = [['#', 'Parameter', 'Severity', 'Confidence', 'Location', 'Description']]
            for i, f in enumerate(result.findings, 1):
                conf_pct = f"{int(f.confidence * 100)}%" if f.confidence > 0 else "N/A"
                rows.append([str(i), f.defect_type, f.severity, conf_pct, f.location, f.description])

            dt = Table(rows, colWidths=[0.3*inch, 1.3*inch, 0.7*inch, 0.7*inch, 1.4*inch, 2.1*inch])
            ts_list = [
                ('BACKGROUND', (0, 0), (-1, 0), self.NAVY),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8.5),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('GRID', (0, 0), (-1, -1), 0.5, self.DIVIDER),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('LEFTPADDING', (0, 0), (-1, -1), 4),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.LIGHT]),
            ]
            sev_color = {'Critical': self.RED, 'Major': self.ORANGE, 'Minor': self.YELLOW}
            for ri in range(1, len(rows)):
                sc = sev_color.get(rows[ri][2])
                if sc:
                    ts_list.append(('TEXTCOLOR', (2, ri), (2, ri), sc))
            dt.setStyle(TableStyle(ts_list))
            elems.append(dt)
        else:
            elems.append(Spacer(1, 8))
            ok = ParagraphStyle('OK', parent=styles['Normal'],
                                 fontSize=11, textColor=self.GREEN)
            elems.append(Paragraph(
                '[PASS] All inspection parameters passed. No visible oil leakage, cracks, cuts, or loose connections.', ok))

        return elems

    # ── Summary Page ──────────────────────────────────────────────────────

    def _summary_page(self, styles, results):
        elems = []

        hdr = ParagraphStyle('SHdr', parent=styles['Title'],
                              fontSize=20, textColor=self.NAVY,
                              spaceAfter=15, alignment=TA_CENTER)
        elems.append(Paragraph('Inspection Batch Summary', hdr))
        elems.append(HRFlowable(width='100%', thickness=2, color=self.NAVY, spaceAfter=20))

        rows = [['Asset ID', 'Type', 'Condition', 'Defects', 'Max Severity', 'Status']]
        for r in results:
            n = len(r.findings)
            sev = 'None'
            status = 'Pass'
            if r.findings:
                status = 'Action Req.'
                order = {'Critical': 3, 'Major': 2, 'Minor': 1}
                sev = max(r.findings, key=lambda f: order.get(f.severity, 0)).severity
            rows.append([r.asset_id, r.asset_type, r.overall_condition, str(n), sev, status])

        st = Table(rows, colWidths=[1*inch, 1.4*inch, 1*inch, 0.7*inch, 1.1*inch, 1.1*inch])
        ss = [
            ('BACKGROUND', (0, 0), (-1, 0), self.NAVY),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, self.DIVIDER),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.LIGHT]),
        ]
        for ri in range(1, len(rows)):
            c = rows[ri][2]
            if c == 'Faulty':
                ss.append(('TEXTCOLOR', (2, ri), (2, ri), self.RED))
                ss.append(('FONTNAME', (2, ri), (2, ri), 'Helvetica-Bold'))
                ss.append(('TEXTCOLOR', (5, ri), (5, ri), self.RED))
            elif c == 'Healthy':
                ss.append(('TEXTCOLOR', (2, ri), (2, ri), self.GREEN))
                ss.append(('TEXTCOLOR', (5, ri), (5, ri), self.GREEN))
        st.setStyle(TableStyle(ss))
        st.hAlign = 'CENTER'
        elems.append(st)

        elems.append(Spacer(1, 25))

        total = len(results)
        faulty = sum(1 for r in results if r.overall_condition == 'Faulty')
        healthy = total - faulty
        total_defects = sum(len(r.findings) for r in results)

        stat = ParagraphStyle('Stat', parent=styles['Normal'],
                               fontSize=11, textColor=colors.HexColor('#333333'), spaceAfter=6)
        elems.append(Paragraph(f'<b>Total Assets Inspected:</b> {total}', stat))
        elems.append(Paragraph(
            f"<b>Healthy Assets:</b> <font color='#27ae60'>{healthy}</font>", stat))
        elems.append(Paragraph(
            f"<b>Faulty Assets:</b> <font color='#c0392b'>{faulty}</font>", stat))
        elems.append(Paragraph(f'<b>Total Defects Identified:</b> {total_defects}', stat))

        elems.append(Spacer(1, 35))
        elems.append(HRFlowable(width='60%', thickness=1, color=self.DIVIDER, spaceAfter=10))

        ft = ParagraphStyle('Ft', parent=styles['Normal'],
                             fontSize=9, textColor=colors.HexColor('#888888'),
                             alignment=TA_CENTER)
        elems.append(Paragraph(
            'Report generated by TCS Autonomous Robot Vision Inspection System', ft))
        elems.append(Paragraph(
            f"Generated on: {datetime.now().strftime('%d %B %Y at %H:%M:%S IST')}", ft))
        elems.append(Paragraph(
            '\u00a9 Tata Consultancy Services Ltd. All rights reserved.', ft))

        return elems
