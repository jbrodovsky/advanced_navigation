import { Presentation, PresentationFile } from "@oai/artifact-tool";

const OUT = new URL("./advanced-autonomous-navigation-lecture-slides.pptx", import.meta.url)
  .pathname;

const WIDTH = 1280;
const HEIGHT = 720;
const MARGIN = 72;
const INK = "#111827";
const MUTED = "#475569";
const FAINT = "#E2E8F0";
const PANEL = "#F8FAFC";
const ACCENT = "#0F766E";

const chapters = [
  {
    part: "Introduction", number: "01", title: "Introduction to Advanced Navigation",
    prompt: "What does this course connect?",
    bullets: ["Classical maritime and aerospace navigation", "Modern robotics: sensing, estimation, and mapping", "Theory paired with runnable Python"],
    source: "book/chapters/00-intro/intro.qmd",
  },
  {
    part: "Introduction", number: "02", title: "Guidance, Navigation, and Control",
    prompt: "What does navigation owe the rest of autonomy?",
    bullets: ["Guidance chooses where to go", "Navigation estimates where the vehicle is", "Control acts on a state with time, frame, and uncertainty"],
    source: "book/chapters/00-intro/navigation-overview.qmd",
  },
  {
    part: "Introduction", number: "03", title: "Syllabus and Course Rhythm",
    prompt: "How will we learn it?",
    bullets: ["Read the notes as the detailed script", "Use code to make mathematical claims testable", "Build from local estimation to SLAM and INS/GNSS"],
    source: "book/chapters/00-intro/syllabus.qmd",
  },
  {
    part: "Foundations", number: "04", title: "Coordinate Frames and Attitude",
    prompt: "Can every vector name its frame?",
    bullets: ["Adopt one explicit convention contract", "Use rotations to move quantities between frames", "Test transforms before trusting them"],
    source: "book/chapters/01-foundations/frames-and-attitude.qmd",
  },
  {
    part: "Foundations", number: "05", title: "Linear Algebra Refresher",
    prompt: "Which operations make estimation work?",
    bullets: ["Vectors and matrices express state and models", "Factorizations beat explicit inverses", "Least squares links measurements to estimates"],
    source: "book/chapters/01-foundations/linear-algebra.qmd",
  },
  {
    part: "Foundations", number: "06", title: "Probability Theory Refresher",
    prompt: "How do we represent what we do not know?",
    bullets: ["Priors, likelihoods, and posteriors", "Covariance captures scale and correlation", "A house edge is just a biased probability model with good marketing"],
    source: "book/chapters/01-foundations/probability-theory.qmd",
  },
  {
    part: "Foundations", number: "07", title: "Modern Python Refresher",
    prompt: "What makes numerical code dependable?",
    bullets: ["Use a reproducible environment and clear structure", "Make array shapes and units explicit", "Plot, test, and fail loudly"],
    source: "book/chapters/01-foundations/modern-python.qmd",
  },
  {
    part: "Sensing and Perception", number: "08", title: "Sensing and Perception",
    prompt: "Why is a sensor not an estimate?",
    bullets: ["Sensors report imperfect physical evidence", "Proprioception and exteroception fail differently", "An estimator turns measurements into a state belief"],
    source: "book/chapters/02-sensing/sensing.qmd",
  },
  {
    part: "Sensing and Perception", number: "09", title: "Motion Models",
    prompt: "How does motion create uncertainty?",
    bullets: ["Kinematics predicts where state may move", "Gyroscopes and accelerometers are biased evidence", "The filter sees calibrated measurements and noise models"],
    source: "book/chapters/02-sensing/motion-models.qmd",
  },
  {
    part: "Sensing and Perception", number: "10", title: "Measurement Noise and Range Models",
    prompt: "What does a realistic observation model explain?",
    bullets: ["Measurements are probabilistic evidence", "A lidar beam has multiple plausible failure modes", "Noise is a modeling choice, not decoration"],
    source: "book/chapters/02-sensing/measurement-models.qmd",
  },
  {
    part: "Sensing and Perception", number: "11", title: "Features, Maps, and Data Association",
    prompt: "Which observation belongs to which feature?",
    bullets: ["Extract useful features from raw signals", "Gate against predicted measurements", "Ambiguity can quietly corrupt an otherwise good filter"],
    source: "book/chapters/02-sensing/data-association.qmd",
  },
  {
    part: "Sensing and Perception", number: "12", title: "Perspective-n-Point Vision",
    prompt: "How do pixels become a pose measurement?",
    bullets: ["Camera geometry maps world points to image points", "PnP solves pose from known correspondences", "Calibration, frame transforms, and covariance matter"],
    source: "book/chapters/02-sensing/pnp-vision.qmd",
  },
  {
    part: "Filtering and State Estimation", number: "13", title: "Filtering and State Estimation",
    prompt: "How do beliefs move and learn?",
    bullets: ["Define the navigation state", "Prediction moves belief through a motion model", "Measurement updates pull belief toward evidence"],
    source: "book/chapters/03-filtering/filtering.qmd",
  },
  {
    part: "Filtering and State Estimation", number: "14", title: "Bayes Filtering",
    prompt: "What is the recursive pattern underneath every filter?",
    bullets: ["Keep a normalized belief over state", "Use the Markov assumption to stay recursive", "Discrete filters make every operation visible"],
    source: "book/chapters/03-filtering/bayes.qmd",
  },
  {
    part: "Filtering and State Estimation", number: "15", title: "Kalman Filtering and the EKF",
    prompt: "When is a Gaussian approximation useful?",
    bullets: ["Linear Gaussian systems yield an exact recursion", "The EKF linearizes nonlinear models locally", "Consistency checks and error states keep it honest"],
    source: "book/chapters/03-filtering/ekf.qmd",
  },
  {
    part: "Filtering and State Estimation", number: "16", title: "The Unscented Kalman Filter",
    prompt: "Can we transform uncertainty without Jacobians?",
    bullets: ["Sigma points represent mean and covariance", "The unscented transform captures nonlinear effects", "Compare robustness, cost, and tuning with the EKF"],
    source: "book/chapters/03-filtering/ukf.qmd",
  },
  {
    part: "Filtering and State Estimation", number: "17", title: "The Information Filter",
    prompt: "What changes when we store precision instead of covariance?",
    bullets: ["Information is inverse uncertainty", "Independent measurements add naturally", "Prediction is the tradeoff"],
    source: "book/chapters/03-filtering/information-filter.qmd",
  },
  {
    part: "Filtering and State Estimation", number: "18", title: "Nonparametric and Particle Filters",
    prompt: "What if one Gaussian is not enough?",
    bullets: ["Grids and particles represent richer beliefs", "Weights encode measurement support", "Effective sample size decides when to resample"],
    source: "book/chapters/03-filtering/pf.qmd",
  },
  {
    part: "Simultaneous Localization and Mapping", number: "19", title: "Mapping for SLAM",
    prompt: "What kind of map does the task need?",
    bullets: ["Occupancy grids model space", "Landmark maps model named features", "Map quality begins with sensible extraction and checks"],
    source: "book/chapters/04-slam/mapping.qmd",
  },
  {
    part: "Simultaneous Localization and Mapping", number: "20", title: "Simultaneous Localization and Mapping",
    prompt: "How can a robot estimate pose and map together?",
    bullets: ["Localization needs a map; mapping needs a pose", "FastSLAM factorizes trajectory and landmark uncertainty", "Graph methods shift the problem toward smoothing"],
    source: "book/chapters/04-slam/slam.qmd",
  },
  {
    part: "Simultaneous Localization and Mapping", number: "21", title: "Data Association for Landmark SLAM",
    prompt: "How do we prevent a wrong match from poisoning the map?",
    bullets: ["Use innovations and Mahalanobis distance", "Gate, birth, and delete landmarks deliberately", "Diagnose associations per particle in FastSLAM"],
    source: "book/chapters/04-slam/data-association.qmd",
  },
  {
    part: "Simultaneous Localization and Mapping", number: "22", title: "Visual Navigation and Visual SLAM",
    prompt: "What carries from camera geometry into SLAM?",
    bullets: ["Features and geometry turn images into motion cues", "Visual odometry and visual SLAM solve different problems", "Visual-inertial systems add complementary constraints"],
    source: "book/chapters/04-slam/visual-navigation.qmd",
  },
  {
    part: "Simultaneous Localization and Mapping", number: "23", title: "Loop Closure and Long-Term Consistency",
    prompt: "How does a map remember it has been here before?",
    bullets: ["Dead reckoning drift is inevitable", "Recognition needs geometric verification", "Graph optimization distributes a loop correction globally"],
    source: "book/chapters/04-slam/loop-closure.qmd",
  },
  {
    part: "Global Scale Navigation", number: "24", title: "Global Navigation and GNSS Measurements",
    prompt: "Why does the Earth change the navigation problem?",
    bullets: ["Global motion needs ECI, ECEF, and local-frame discipline", "GNSS supplies position, velocity, timing, and uncertainty", "GNSS bounds drift; INS bridges outages and dynamics"],
    source: "book/chapters/05-global/global-nav.qmd",
  },
  {
    part: "Global Scale Navigation", number: "25", title: "Curved-Earth INS and Loose GNSS Integration",
    prompt: "How does an error-state INS use GNSS?",
    bullets: ["Propagate attitude, velocity, and position on WGS84", "Estimate small errors around a nominal trajectory", "Apply time-aligned GNSS updates with frame discipline"],
    source: "book/chapters/05-global/ins-gnss.qmd",
  },
  {
    part: "Global Scale Navigation", number: "26", title: "INS Error Models and IMU Characterization",
    prompt: "Where does inertial drift come from?",
    bullets: ["Bias, scale, noise, temperature, and vibration all matter", "Allan deviation exposes stochastic behavior", "Sensor evidence becomes process noise and state design"],
    source: "book/chapters/05-global/ins-errors.qmd",
  },
  {
    part: "Practical Implementation Guide", number: "27", title: "From Prototype to Vehicle Software",
    prompt: "When is an estimator ready to leave the notebook?",
    bullets: ["Prove correctness on replay before real time", "Treat timing, frames, health, and logs as contracts", "Validate through progressively less forgiving environments"],
    source: "book/chapters/06-practical/practical-implementation-guide.qmd",
  },
  {
    part: "Advanced Topics", number: "28", title: "Factor Graphs, Manifolds, and Modern Back-ends",
    prompt: "What lies beyond the core course?",
    bullets: ["Factor graphs, visual-inertial methods, and resilient navigation", "Hybrid and learning-based methods extend the same foundations", "Read research by checking assumptions and evidence"],
    source: "book/chapters/07-adv/advanced-slam.qmd",
  },
  {
    part: "Advanced Topics", number: "29", title: "GNSS Signals, Receivers, and Deep Integration",
    prompt: "What do we gain by moving closer to the receiver?",
    bullets: ["Signals become pseudorange, Doppler, and carrier observations", "Coupling depth changes the estimator boundary", "Accuracy, integrity, and resilience are separate concerns"],
    source: "book/chapters/07-adv/adv-gnss.qmd",
  },
  {
    part: "Advanced Topics", number: "30", title: "Navigation System Integration Case Study",
    prompt: "What must a deployed estimator promise?",
    bullets: ["Separate transport, calibration, estimation, health, and logging", "Update at measurement time, not arrival time", "Publish state with uncertainty, validity time, and mode"],
    source: "book/chapters/06-practical/practical-implementation-guide.qmd",
  },
  {
    part: "Technical Reference", number: "31", title: "Mathematical Reference and Derivations",
    prompt: "Where do we go when the short version is not enough?",
    bullets: ["Kalman, EKF, quaternion, and optimization derivations", "Numerical stability guidance", "A reference chapter: consult it when implementation needs proof"],
    source: "book/chapters/01-foundations/mathematical-reference.qmd",
  },
];

const parts = [
  ["Introduction", "01", "Why estimation is the center of autonomy"],
  ["Foundations", "02", "Frames, linear algebra, probability, and Python"],
  ["Sensing and Perception", "03", "Turning physical evidence into useful measurements"],
  ["Filtering and State Estimation", "04", "Representing and updating belief"],
  ["Simultaneous Localization and Mapping", "05", "Learning a map while locating the vehicle"],
  ["Global Scale Navigation", "06", "Inertial navigation, GNSS, and a rotating Earth"],
  ["Practical Implementation Guide", "07", "Making the estimator reproducible and usable"],
  ["Advanced Topics", "08", "Where the core ideas extend next"],
  ["Technical Reference", "09", "Optional derivations when the short version is not enough"],
];

function textbox(slide, name, text, position, style = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox", name, position, fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    typeface: "Helvetica Neue", color: INK, fontSize: 24,
    verticalAlignment: "top", autoFit: "none", wrap: "square",
    insets: { top: 0, right: 0, bottom: 0, left: 0 }, ...style,
  };
  return shape;
}

function rule(slide, top, left = MARGIN, width = WIDTH - 2 * MARGIN, color = FAINT, height = 2) {
  return slide.shapes.add({
    geometry: "rect", name: "rule", position: { left, top, width, height },
    fill: color, line: { style: "solid", fill: color, width: 0 },
  });
}

function footer(slide, number, label = "Advanced Robot Navigation") {
  textbox(slide, "footer-label", label, { left: MARGIN, top: 666, width: 800, height: 20 },
    { fontSize: 14, color: MUTED });
  textbox(slide, "footer-number", String(number).padStart(2, "0"),
    { left: 1160, top: 662, width: 48, height: 24 },
    { fontSize: 16, bold: true, color: MUTED, alignment: "right" });
}

function notes(slide, source) {
  slide.speakerNotes.textFrame.setText([
    "The detailed presenter script is the corresponding course-book chapter.",
    `Book source: ${source}`,
  ]);
  slide.speakerNotes.setVisible(true);
}

function addCover(deck) {
  const slide = deck.slides.add();
  slide.background.fill = "#FFFFFF";
  textbox(slide, "eyebrow", "GRADUATE COURSE LECTURE SLIDES", { left: MARGIN, top: 72, width: 550, height: 28 },
    { fontSize: 18, bold: true, color: ACCENT });
  textbox(slide, "title", "Advanced\nAutonomous Navigation", { left: MARGIN, top: 180, width: 940, height: 220 },
    { fontSize: 72, bold: true, lineSpacing: 0.92 });
  textbox(slide, "subtitle", "Lightweight lecture prompts for a course in state estimation, SLAM, and inertial navigation", { left: MARGIN, top: 472, width: 700, height: 76 },
    { fontSize: 27, color: MUTED, lineSpacing: 1.15 });
  rule(slide, 610, MARGIN, 460, ACCENT, 5);
  textbox(slide, "byline", "The book is the script; these slides keep the room oriented.", { left: MARGIN, top: 634, width: 800, height: 30 },
    { fontSize: 18, color: MUTED });
  notes(slide, "book/_quarto.yml");
}

function addOrientation(deck, number) {
  const slide = deck.slides.add();
  textbox(slide, "title", "Use the slides to navigate; use the book to learn.", { left: MARGIN, top: 68, width: 1040, height: 62 },
    { fontSize: 42, bold: true });
  rule(slide, 160);
  const cols = [
    ["Before class", "Skim the chapter and identify the central question."],
    ["During class", "Use these prompts to follow the argument, examples, and code."],
    ["After class", "Return to the notebook-like notes for derivations and exercises."],
  ];
  cols.forEach(([head, body], i) => {
    const left = MARGIN + i * 384;
    textbox(slide, `orientation-head-${i}`, `0${i + 1}`, { left, top: 244, width: 100, height: 46 },
      { fontSize: 24, bold: true, color: ACCENT });
    textbox(slide, `orientation-title-${i}`, head, { left, top: 310, width: 310, height: 46 },
      { fontSize: 29, bold: true });
    textbox(slide, `orientation-body-${i}`, body, { left, top: 378, width: 310, height: 118 },
      { fontSize: 23, color: MUTED, lineSpacing: 1.2 });
  });
  footer(slide, number);
  notes(slide, "book/_quarto.yml");
}

function addSection(deck, number, ordinal, part, subtitle) {
  const slide = deck.slides.add();
  slide.background.fill = "#F8FAFC";
  textbox(slide, "section-number", ordinal, { left: MARGIN, top: 82, width: 120, height: 44 },
    { fontSize: 26, bold: true, color: ACCENT });
  textbox(slide, "section-title", part, { left: MARGIN, top: 228, width: 1010, height: 130 },
    { fontSize: 62, bold: true, lineSpacing: 0.95 });
  textbox(slide, "section-subtitle", subtitle, { left: MARGIN, top: 430, width: 760, height: 80 },
    { fontSize: 29, color: MUTED, lineSpacing: 1.2 });
  rule(slide, 584, MARGIN, 420, ACCENT, 5);
  footer(slide, number, part);
  notes(slide, "book/_quarto.yml");
}

function bulletText(items) {
  return items.map((item) => ({
    bulletCharacter: "•", marginLeft: 30, indent: -18, spaceAfter: 18,
    runs: [item],
  }));
}

function addChapter(deck, number, chapter, variant) {
  const slide = deck.slides.add();
  const titleHeight = chapter.title.length > 44 ? 106 : 64;
  textbox(slide, "chapter-part", chapter.part.toUpperCase(), { left: MARGIN, top: 54, width: 840, height: 28 },
    { fontSize: 16, bold: true, color: ACCENT });
  textbox(slide, "chapter-title", chapter.title, { left: MARGIN, top: 98, width: 1030, height: titleHeight },
    { fontSize: 42, bold: true, lineSpacing: 0.98 });
  rule(slide, 98 + titleHeight + 22);

  if (variant === 0) {
    textbox(slide, "question-label", "CORE QUESTION", { left: MARGIN, top: 265, width: 280, height: 24 },
      { fontSize: 16, bold: true, color: ACCENT });
    textbox(slide, "question", chapter.prompt, { left: MARGIN, top: 310, width: 334, height: 180 },
      { fontSize: 31, bold: true, lineSpacing: 1.05 });
    rule(slide, 304, 452, 2, ACCENT, 290);
    const bullets = textbox(slide, "bullets", bulletText(chapter.bullets), { left: 510, top: 274, width: 650, height: 300 },
      { fontSize: 26, color: INK, lineSpacing: 1.12 });
    bullets.text.style = { ...bullets.text.style, fontSize: 26 };
  } else if (variant === 1) {
    textbox(slide, "prompt", chapter.prompt, { left: MARGIN, top: 244, width: 950, height: 56 },
      { fontSize: 31, bold: true, color: ACCENT });
    chapter.bullets.forEach((item, i) => {
      const left = MARGIN + i * 384;
      textbox(slide, `step-${i}`, String(i + 1).padStart(2, "0"), { left, top: 360, width: 64, height: 34 },
        { fontSize: 18, bold: true, color: MUTED });
      textbox(slide, `idea-${i}`, item, { left, top: 416, width: 320, height: 122 },
        { fontSize: 24, bold: true, lineSpacing: 1.15 });
    });
  } else {
    textbox(slide, "prompt", chapter.prompt, { left: MARGIN, top: 242, width: 950, height: 56 },
      { fontSize: 31, bold: true, color: ACCENT });
    const bullets = textbox(slide, "bullets", bulletText(chapter.bullets), { left: MARGIN, top: 346, width: 1010, height: 230 },
      { fontSize: 27, color: INK, lineSpacing: 1.12 });
    bullets.text.style = { ...bullets.text.style, fontSize: 27 };
  }
  footer(slide, number, chapter.part);
  notes(slide, chapter.source);
}

function addClosing(deck, number) {
  const slide = deck.slides.add();
  textbox(slide, "closing-eyebrow", "THE COURSE HABIT", { left: MARGIN, top: 90, width: 440, height: 28 },
    { fontSize: 18, bold: true, color: ACCENT });
  textbox(slide, "closing-title", "Make every navigation claim testable.", { left: MARGIN, top: 220, width: 950, height: 78 },
    { fontSize: 54, bold: true });
  textbox(slide, "closing-body", "Name the frame. State the time. Carry the uncertainty. Keep the replay.", { left: MARGIN, top: 360, width: 850, height: 80 },
    { fontSize: 31, color: MUTED, lineSpacing: 1.15 });
  rule(slide, 552, MARGIN, 460, ACCENT, 5);
  footer(slide, number);
  notes(slide, "book/chapters/06-practical/practical-implementation-guide.qmd");
}

async function main() {
  const deck = Presentation.create({ slideSize: { width: WIDTH, height: HEIGHT } });
  addCover(deck);
  let slideNumber = 2;
  addOrientation(deck, slideNumber++);
  let partIndex = 0;
  for (const [part, ordinal, subtitle] of parts) {
    addSection(deck, slideNumber++, ordinal, part, subtitle);
    chapters.filter((chapter) => chapter.part === part).forEach((chapter, i) => {
      addChapter(deck, slideNumber++, chapter, (i + partIndex) % 3);
    });
    partIndex += 1;
  }
  addClosing(deck, slideNumber++);
  const pptx = await PresentationFile.exportPptx(deck);
  await pptx.save(OUT);
  console.log(`Wrote ${OUT} with ${deck.slides.items.length} slides.`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
