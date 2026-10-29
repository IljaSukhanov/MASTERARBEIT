from pathlib import Path
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path("/Users/iljasukhanov/MASTERARBEIT")
OUT = ROOT / "output/pdf/minimum_width_linf_beta_experiment.pdf"
TMP = ROOT / "tmp/pdfs/minimum_width_note"
OUT.parent.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)

FONT_DIR = Path(
    "/Users/iljasukhanov/.cache/codex-runtimes/codex-primary-runtime/"
    "dependencies/native/libreoffice-headless/libreoffice/LibreOfficeDev.app/"
    "Contents/Resources/fonts/truetype"
)
pdfmetrics.registerFont(TTFont("DVSans", str(FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DVSans-Bold", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DVSerif", str(FONT_DIR / "DejaVuSerif.ttf")))
pdfmetrics.registerFont(TTFont("DVSerif-Italic", str(FONT_DIR / "DejaVuSerif-Italic.ttf")))
pdfmetrics.registerFont(TTFont("DVSerif-Bold", str(FONT_DIR / "DejaVuSerif-Bold.ttf")))

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2E6F95")
PALE_BLUE = colors.HexColor("#EAF3F8")
PALE_GOLD = colors.HexColor("#FBF4E4")
GOLD = colors.HexColor("#B88B2A")
INK = colors.HexColor("#1F2933")
MUTED = colors.HexColor("#5B6770")
RULE = colors.HexColor("#D6DEE3")
WHITE = colors.white


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="TitleCustom",
    fontName="DVSans-Bold",
    fontSize=25,
    leading=30,
    textColor=NAVY,
    spaceAfter=5 * mm,
))
styles.add(ParagraphStyle(
    name="SubtitleCustom",
    fontName="DVSerif-Italic",
    fontSize=12,
    leading=17,
    textColor=MUTED,
    spaceAfter=8 * mm,
))
styles.add(ParagraphStyle(
    name="HeadingCustom",
    fontName="DVSans-Bold",
    fontSize=15,
    leading=19,
    textColor=NAVY,
    spaceBefore=5 * mm,
    spaceAfter=2.5 * mm,
))
styles.add(ParagraphStyle(
    name="BodyCustom",
    fontName="DVSerif",
    fontSize=10.5,
    leading=15.2,
    textColor=INK,
    spaceAfter=2.8 * mm,
    allowWidows=0,
    allowOrphans=0,
))
styles.add(ParagraphStyle(
    name="BodyNoAfter",
    parent=styles["BodyCustom"],
    spaceAfter=0,
))
styles.add(ParagraphStyle(
    name="Lead",
    parent=styles["BodyCustom"],
    fontSize=11.2,
    leading=16.2,
    textColor=NAVY,
    spaceAfter=4 * mm,
))
styles.add(ParagraphStyle(
    name="Callout",
    fontName="DVSans-Bold",
    fontSize=10.8,
    leading=15,
    alignment=TA_CENTER,
    textColor=NAVY,
))
styles.add(ParagraphStyle(
    name="Quote",
    fontName="DVSerif-Italic",
    fontSize=10.4,
    leading=14.7,
    textColor=INK,
    leftIndent=3 * mm,
    rightIndent=3 * mm,
))
styles.add(ParagraphStyle(
    name="TableHead",
    fontName="DVSans-Bold",
    fontSize=8.7,
    leading=11,
    textColor=WHITE,
    alignment=TA_CENTER,
))
styles.add(ParagraphStyle(
    name="TableCell",
    fontName="DVSerif",
    fontSize=8.8,
    leading=11.5,
    textColor=INK,
    alignment=TA_CENTER,
))
styles.add(ParagraphStyle(
    name="Ref",
    fontName="DVSerif",
    fontSize=8.7,
    leading=12,
    textColor=MUTED,
    leftIndent=4 * mm,
    firstLineIndent=-4 * mm,
))


def para(text, style="BodyCustom"):
    return Paragraph(text, styles[style])


def equation(tex, width_mm=None, fontsize=17, pad=2.0):
    """Render a display equation using MathText and return a Platypus image."""
    fig = plt.figure(figsize=(1, 1), dpi=220)
    fig.patch.set_alpha(0)
    txt = fig.text(
        0.5,
        0.5,
        f"${tex}$",
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#17324D",
        math_fontfamily="stix",
    )
    fig.canvas.draw()
    bbox = txt.get_window_extent(renderer=fig.canvas.get_renderer()).expanded(1.05, 1.24)
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=220,
        transparent=True,
        bbox_inches=bbox.transformed(fig.dpi_scale_trans.inverted()),
        pad_inches=pad / 25.4,
    )
    plt.close(fig)
    buf.seek(0)
    img = Image(buf)
    if width_mm:
        ratio = img.imageHeight / img.imageWidth
        img.drawWidth = width_mm * mm
        img.drawHeight = img.drawWidth * ratio
    else:
        max_w = 155 * mm
        scale = min(1.0, max_w / img.imageWidth)
        img.drawWidth = img.imageWidth * scale
        img.drawHeight = img.imageHeight * scale
    img.hAlign = "CENTER"
    return img


def eq_block(tex, width_mm=None, fontsize=17, before=1.2, after=2.5):
    return KeepTogether([
        Spacer(1, before * mm),
        equation(tex, width_mm=width_mm, fontsize=fontsize),
        Spacer(1, after * mm),
    ])


def callout(contents, background=PALE_BLUE, border=BLUE):
    if not isinstance(contents, list):
        contents = [contents]
    table = Table([[contents]], colWidths=[164 * mm], hAlign="CENTER")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), background),
        ("BOX", (0, 0), (-1, -1), 0.8, border),
        ("LEFTPADDING", (0, 0), (-1, -1), 7 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 4 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4 * mm),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return table


def on_page(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(23 * mm, height - 16 * mm, width - 23 * mm, height - 16 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont("DVSans", 7.7)
    canvas.drawString(23 * mm, height - 12.5 * mm, "MINIMUM WIDTH AND L-INFINITY REGRESSION")
    canvas.drawRightString(width - 23 * mm, 11.5 * mm, f"{doc.page}")
    canvas.restoreState()


doc = BaseDocTemplate(
    str(OUT),
    pagesize=A4,
    leftMargin=23 * mm,
    rightMargin=23 * mm,
    topMargin=23 * mm,
    bottomMargin=19 * mm,
    title="Minimum-Width Estimation, L-Infinity Regression, and Beta Innovations",
    subject="A methodological note on bounded-noise estimators",
    creator="Compiled from user-supplied text",
)
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=on_page)])


story = []
story += [
    Spacer(1, 11 * mm),
    para("Minimum-Width Estimation, <i>L</i><super>∞</super> Regression, and Beta Innovations", "TitleCustom"),
    para("An important equivalence, its limits, and what a Beta experiment can reveal", "SubtitleCustom"),
    callout([
        para(
            "<b>Correction in one sentence.</b> The minimum-width estimator is not merely another bounded-noise estimator: it is exactly twice <i>L</i><super>∞</super> regression when the center (or intercept) is optimized.",
            "Lead",
        )
    ]),
    Spacer(1, 7 * mm),
    para(
        "Yes - but there is an important correction to my previous suggestion: <b>your minimum-width estimator and <i>L</i><super>∞</super> regression are much more closely related than \"two competing bounded-noise estimators.\"</b>",
        "Lead",
    ),
    para("For your AR residuals"),
    eq_block(r"r_t(\phi)=X_t-\sum_{j=1}^{p}\phi_jX_{t-j},", width_mm=58),
    para("your minimum-width criterion is essentially"),
    eq_block(r"W_n(\phi)=\max_t r_t(\phi)-\min_t r_t(\phi),", width_mm=76),
    para("whereas ordinary zero-centered <i>L</i><super>∞</super> regression uses"),
    eq_block(r"Q_n(\phi)=\max_t\left|r_t(\phi)\right|.", width_mm=52),
    para("So they <b>can differ</b>. But observe the identity"),
    eq_block(
        r"\min_{c\in\mathbb{R}}\max_t\left|r_t(\phi)-c\right|"
        r"=\frac{1}{2}\left(\max_t r_t(\phi)-\min_t r_t(\phi)\right).",
        width_mm=122,
        fontsize=16,
    ),
    callout([
        para("<b>minimum width</b> = 2 × <i>L</i><super>∞</super> regression with an optimally chosen center/intercept.", "Callout")
    ]),
    Spacer(1, 3.5 * mm),
    para("That completely changes how I would think about the Beta experiment."),
]


story += [
    para("Where Beta innovations become interesting", "HeadingCustom"),
    para("Suppose first that"),
    eq_block(r"\varepsilon_t\sim\operatorname{Unif}[-a,a].", width_mm=48),
    para(
        "The distribution is symmetric around zero. Zero-centered <i>L</i><super>∞</super> is perfectly natural, and your width criterion has little extra information to exploit."
    ),
    para("Now suppose instead you take a transformed Beta distribution. On [−<i>a</i>, <i>a</i>],"),
    eq_block(
        r"Z=\frac{\varepsilon+a}{2a}\sim\operatorname{Beta}(\alpha,\beta).",
        width_mm=72,
    ),
    para("Its density is proportional to"),
    eq_block(
        r"(\varepsilon+a)^{\alpha-1}(a-\varepsilon)^{\beta-1}.",
        width_mm=72,
    ),
    para(
        "For α = β = 1, this is uniform, and we recover the situation above. For α = β = 2, the density is concentrated toward the middle and vanishes at the boundaries. For α = β &lt; 1, it is U-shaped and puts much more mass near the two boundaries. And for α ≠ β, it is asymmetric."
    ),
    para(
        "The <b>correct Beta likelihood knows all of this</b>. Neither your minimum-width criterion nor <i>L</i><super>∞</super> does. A correctly specified Beta likelihood contains terms involving all residuals, not merely their extremes. For fixed support, it looks schematically like"
    ),
    eq_block(
        r"\ell(\phi)\propto(\alpha-1)\sum_t\log(r_t(\phi)-L)"
        r"+(\beta-1)\sum_t\log(U-r_t(\phi)).",
        width_mm=124,
        fontsize=15.5,
    ),
    para(
        "So I would expect a correctly specified Beta MLE generally to be the benchmark that is hardest to beat when the Beta model is actually true. Estimation of unknown support endpoints makes such models nonregular and introduces additional subtleties; the literature explicitly notes unusual behavior for unknown-support Beta models. <a href=\"https://www.sciencedirect.com/science/article/pii/S0167947321001791\" color=\"#2E6F95\">[1]</a>"
    ),
]


story += [
    para("But can minimum width beat ordinary L-infinity?", "HeadingCustom"),
    para("<b>Yes.</b> And there is a clear mechanism."),
    para("Suppose that two candidate values of φ produce the following residual extremes:"),
]

comparison = Table(
    [
        [
            para("Candidate", "TableHead"),
            para("min r<sub>t</sub>", "TableHead"),
            para("max r<sub>t</sub>", "TableHead"),
            para("Q<sub>n</sub>(φ)", "TableHead"),
            para("W<sub>n</sub>(φ)", "TableHead"),
        ],
        [para("First", "TableCell"), para("−1", "TableCell"), para("3", "TableCell"), para("3", "TableCell"), para("4", "TableCell")],
        [para("Second", "TableCell"), para("−2.2", "TableCell"), para("2.2", "TableCell"), para("2.2", "TableCell"), para("4.4", "TableCell")],
    ],
    colWidths=[38 * mm, 28 * mm, 28 * mm, 32 * mm, 32 * mm],
    hAlign="CENTER",
)
comparison.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), NAVY),
    ("BACKGROUND", (0, 1), (-1, 1), PALE_BLUE),
    ("BACKGROUND", (0, 2), (-1, 2), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.5, RULE),
    ("TOPPADDING", (0, 0), (-1, -1), 3 * mm),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3 * mm),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
]))
story += [
    comparison,
    Spacer(1, 4 * mm),
    para(
        "Thus <i>L</i><super>∞</super> prefers the second candidate because its residuals are better centered around <b>zero</b>: 2.2 &lt; 3. Your criterion prefers the first because its residual cloud is genuinely <b>narrower</b>: 4 &lt; 4.4."
    ),
    para("Geometrically that is exactly your distinction:"),
    callout([
        para("<b><i>L</i><super>∞</super>:</b> narrowest slab forced to be centered appropriately around zero", "Quote"),
        Spacer(1, 2.5 * mm),
        para("<b>Minimum width:</b> narrowest affine slab, with its center free", "Quote"),
    ], background=PALE_GOLD, border=GOLD),
    Spacer(1, 3.5 * mm),
    para("That difference can matter especially for <b>asymmetric bounded innovations</b>."),
]


story += [
    para("But here's the catch", "HeadingCustom"),
    para(
        "If you compare your method against <i>L</i><super>∞</super> regression <b>with an intercept/location parameter</b>, then the distinction disappears:"
    ),
    eq_block(
        r"\min_{\phi,c}\max_t\left|r_t(\phi)-c\right|"
        r"=\frac{1}{2}\min_{\phi}W_n(\phi).",
        width_mm=92,
    ),
    para(
        "So you cannot reasonably claim <i>\"Minimum width beats <i>L</i><super>∞</super> because Beta innovations are asymmetric\"</i> if the <i>L</i><super>∞</super> competitor is allowed to estimate the center as well. They are then mathematically the same estimator for φ."
    ),
    para(
        "This is actually a useful realization for your thesis rather than a disappointment. It tells you very precisely what is novel and what is not."
    ),
    callout([
        para(
            "Your minimum-width formulation is a <b>geometric generalization and reinterpretation of Chebyshev regression</b> in which the affine slab is allowed to move.",
            "Callout",
        )
    ]),
    Spacer(1, 3.5 * mm),
    para(
        "Its larger potential advantage may lie elsewhere - especially in the <b>dimension/order-detection geometry</b> you have been developing, rather than in coefficient estimation alone."
    ),
]


story += [
    para("A revealing Beta simulation", "HeadingCustom"),
    para("Your Beta simulations could still be quite revealing. A particularly informative experiment would vary"),
    eq_block(
        r"\operatorname{Beta}(\alpha,\beta)\quad\text{through}\quad"
        r"(1,1),\ (2,2),\ (1/2,1/2),\ (2,5),",
        width_mm=108,
        fontsize=15.5,
    ),
    para("and compare"),
    eq_block(
        r"\mathrm{LS},\quad L^\infty\ \mathrm{(zero\!\!-\!centered)},\quad"
        r"\mathrm{minimum\ width},\quad\mathrm{Beta\ MLE}.",
        width_mm=128,
        fontsize=15,
    ),
    para(
        "The results would tell you <b>why</b> an estimator works: Gaussian-type average information, fixed-center extremes, free-center width geometry, or full knowledge of the innovation distribution. That would be much more scientifically interesting than merely demonstrating that <i>L</i><super>∞</super> beats LS under uniform noise."
    ),
    Spacer(1, 5 * mm),
    para("Reference", "HeadingCustom"),
    para(
        "[1] <a href=\"https://www.sciencedirect.com/science/article/pii/S0167947321001791\" color=\"#2E6F95\"><i>Bayesian beta regression for bounded responses with unknown supports</i></a>, ScienceDirect.",
        "Ref",
    ),
]


doc.build(story)
print(OUT)
