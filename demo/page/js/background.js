let degree = 0; // Initialize degree
let currentColorIndexStart = 0;
let currentColorIndexEnd = 1;
let transitionProgress = 0; // For color interpolation progress
const transitionSpeed = 0.005; // Adjust this for slower/faster color transition
const bodyStyle = document.body.style;

// Declare colors array outside the condition
let colors = []; // Initialize as an empty array

const root = document.documentElement;

if (window.location.pathname.includes('/hidden')) {
    root.style.setProperty('--bg-color', '#ccc5');
    root.style.setProperty('--bg-light-color', '#0004');
    root.style.setProperty('--text-color', '#000');
    root.style.setProperty('--border-color', '#0004');
    root.style.setProperty('--highlight-color', '#0005');
    root.style.setProperty('--darker-highlight-color', '#0007');
    
    colors = [
        [255, 173, 173],
        [255, 214, 165],
        [253, 255, 182],
        [202, 255, 191],
        [155, 246, 255],
        [160, 196, 255],
        [188, 178, 255],
        [255, 198, 255]
    ];
} else {
    root.style.setProperty('--bg-color', '#0005');
    root.style.setProperty('--bg-light-color', '#fff4');
    root.style.setProperty('--text-color', '#fff');
    root.style.setProperty('--border-color', '#fff4');
    root.style.setProperty('--highlight-color', '#0005');
    root.style.setProperty('--darker-highlight-color', '#0007');
    
    colors = [
        [81, 19, 19],
        [84, 57, 30],
        [79, 68, 22],
        [15, 64, 31],
        [23, 20, 56],
        [33, 33, 33],
        [0, 0, 0]
    ];
}

let prevStartColor = colors[0];
let nextStartColor = colors[1];
let prevEndColor = colors[2];
let nextEndColor = colors[3];

function interpolateColor(color1, color2, factor) {
    return color1.map((c, i) => Math.round(c + factor * (color2[i] - c)));
}

function rgbToCss(rgb) {
    return `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, 1)`;
}

function pickNewRandomColors() {
    let currentColorIndexStart = Math.floor(Math.random() * colors.length);
    let currentColorIndexEnd;

    do {
        currentColorIndexEnd = Math.floor(Math.random() * colors.length);
    } while (currentColorIndexEnd === currentColorIndexStart);

    prevStartColor = nextStartColor;
    prevEndColor = nextEndColor;

    nextStartColor = colors[currentColorIndexStart];
    nextEndColor = colors[currentColorIndexEnd];
}

function updateGradientColors() {
    const interpolatedStartColor = interpolateColor(prevStartColor, nextStartColor, transitionProgress);
    const interpolatedEndColor = interpolateColor(prevEndColor, nextEndColor, transitionProgress);

    document.body.style.setProperty('--bg-gradient-start', rgbToCss(interpolatedStartColor));
    document.body.style.setProperty('--bg-gradient-end', rgbToCss(interpolatedEndColor));

    transitionProgress += transitionSpeed;

    if (transitionProgress >= 1) {
        pickNewRandomColors();
        transitionProgress = 0;
    }
}

function rotateGradient() {
    degree = (degree + 4) % 360;
    document.body.style.background = `linear-gradient(${degree}deg, var(--bg-gradient-start), var(--bg-gradient-end))`;
}

setInterval(() => {
    rotateGradient();
    updateGradientColors();
}, 100);

pickNewRandomColors();