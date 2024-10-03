let degree = 0; // Initialize degree
let currentColorIndexStart = 0;
let currentColorIndexEnd = 1;
let transitionProgress = 0; // For color interpolation progress
const transitionSpeed = 0.005; // Adjust this for slower/faster color transition

let colors = [
    [81,19,19],
    [84,57,30],
    [79,68,22],
    [15,64,31],
    [23,20,56],
    [33, 33, 33],
    [0, 0, 0]
];

let prevStartColor = colors[0]; // Start with initial color
let nextStartColor = colors[1]; // Set next target color
let prevEndColor = colors[2];
let nextEndColor = colors[3];

// Function to interpolate between two colors
function interpolateColor(color1, color2, factor) {
    return color1.map((c, i) => Math.round(c + factor * (color2[i] - c)));
}

// Function to convert RGB array to CSS rgba string
function rgbToCss(rgb) {
    return `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, 1)`;
}

// Function to pick new random colors ensuring they are different
function pickNewRandomColors() {
    currentColorIndexStart = Math.floor(Math.random() * colors.length);
    
    do {
        currentColorIndexEnd = Math.floor(Math.random() * colors.length);
    } while (currentColorIndexEnd === currentColorIndexStart); // Ensure the start and end colors are different

    prevStartColor = nextStartColor;
    prevEndColor = nextEndColor;
    
    nextStartColor = colors[currentColorIndexStart];
    nextEndColor = colors[currentColorIndexEnd];
}

// Function to update gradient colors smoothly
function updateGradientColors() {
    // Interpolating between previous and next color gradually
    const interpolatedStartColor = interpolateColor(prevStartColor, nextStartColor, transitionProgress);
    const interpolatedEndColor = interpolateColor(prevEndColor, nextEndColor, transitionProgress);

    // Update CSS variables with interpolated colors
    document.body.style.setProperty('--bg-gradient-start', rgbToCss(interpolatedStartColor));
    document.body.style.setProperty('--bg-gradient-end', rgbToCss(interpolatedEndColor));

    // Increment transition progress
    transitionProgress += transitionSpeed;

    // Once transition is complete, pick new random colors and reset progress
    if (transitionProgress >= 1) {
        pickNewRandomColors();
        transitionProgress = 0;
    }
}

// Function to rotate the gradient
function rotateGradient() {
    degree = (degree + 1) % 360; // Increment by 1 degree
    document.body.style.background = `linear-gradient(${degree}deg, var(--bg-gradient-start), var(--bg-gradient-end))`;
}

// Combine both rotation and color transition into one interval
setInterval(() => {
    rotateGradient();
    updateGradientColors();
}, 50); // Adjust this for smoother or slower effects

// Initial random colors
pickNewRandomColors();
