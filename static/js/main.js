const existingImages = new Map(); // Store existing images with their seeds as keys
let isGeneratingNewImages = false;
let pendingUpdates = false;

document.getElementById('generateForm').addEventListener('submit', function (event) {
    event.preventDefault();

    // Set the flag to true when a new generation starts
    isGeneratingNewImages = true;

    // Clear existing images for new generation
    existingImages.clear();
    document.getElementById('images').innerHTML = '';

    const formData = new FormData(this);
    fetch('/generate', {
        method: 'POST',
        body: formData
    })
    .then((response) => response.json())
    .then((data) => {
        isGeneratingNewImages = false;
    })
    .catch((error) => {
        console.error('Error:', error);
        isGeneratingNewImages = false; // Reset the flag in case of an error
    });
});

setInterval(() => {
    if (document.visibilityState === 'visible' && !pendingUpdates) {
        pendingUpdates = true; // Prevent overlapping updates

        fetch('/status', { cache: 'no-store' })
            .then((response) => response.json())
            .then((data) => {
                document.getElementById('all').style.display = 'flex';

                updateProgressBars(data);

                processImageUpdates(data.images);
                if (data.images.length < existingImages.size) {
                    existingImages.clear();
                    document.getElementById('images').innerHTML = '';
                }
            })
            .catch((error) => {
                console.error('Error fetching status:', error);
                document.getElementById('all').style.display = 'none';
            })
            .finally(() => {
                pendingUpdates = false; // Allow the next update
            });
    }
}, 2500); // Check every 2.5 seconds

function updateProgressBars(data) {
    const progressText = document.getElementById('progress');
    const dynamicProgressBar = document.getElementById('dynamic-progress-bar');
    const alldynamicProgressBar = document.getElementById('all-dynamic-progress-bar');

    // Update progress value smoothly
    if (Number.isInteger(data.imgprogress)) {
        dynamicProgressBar.style.width = `calc(${data.imgprogress}%)`;
        progressText.innerHTML = `Progress: ${data.imgprogress}% Remaining: ${data.remainingimages}`;
    } else {
        dynamicProgressBar.style.width = `0%`;
        alldynamicProgressBar.style.width = `0%`;
        progressText.innerHTML = `Progress: ${data.imgprogress}`;
    }

    if (Number.isInteger(data.allpercentage)) {
        alldynamicProgressBar.style.width = `calc(${data.allpercentage}%)`;
    } else {
        alldynamicProgressBar.style.width = `0%`;
    }
}

function processImageUpdates(images) {
    const imagesDiv = document.getElementById('images');

    images.forEach((imgData) => {
        const key = imgData.seed; // Assuming seed is the unique identifier for the image

        if (existingImages.has(key)) {
            // Update existing image only if its source has changed
            const existingWrapper = existingImages.get(key);
            const existingImg = existingWrapper.querySelector('img');

            if (existingImg.src !== imgData.img + "?size=small") {
                existingImg.src = imgData.img + "?size=small";
            }
        } else {
            // Create new image element if it doesn't exist
            const wrapper = document.createElement('div');
            wrapper.className = 'image-wrapper';

            const img = document.createElement('img');
            img.src = imgData.img + "?size=small";
            img.loading = "lazy";

            img.onclick = () => {
                openLink("image/" + imgData.img.split('/').pop()); // Send to "image/" + filename
            };

            wrapper.appendChild(img); // Add image to wrapper
            imagesDiv.appendChild(wrapper); // Add wrapper to imagesDiv
            existingImages.set(key, wrapper); // Store the wrapper in the map
        }
    });
}

function openLink(link) {
    window.location.href = link.replace("?size=small", "");
}

document.addEventListener('visibilitychange', function () {
    const state =
        document.visibilityState === 'visible'
            ? 'Vis'
            : document.visibilityState === 'hidden'
            ? 'Hid'
            : document.visibilityState === 'prerender'
            ? 'Pre'
            : 'Unk';
    document.title = `Image Generator (${state})`;
});

document.getElementById('stopButton').addEventListener('click', function() {
    fetch('/stop', {
        method: 'POST'
    })
    .catch(error => console.error('Error stopping generation:', error));
});

document.getElementById('restartButton').addEventListener('click', function() {
    const isConfirmed = confirm('Are you sure you want to restart the server?\nIt will reset all of the variables and has a chance to fail restarting');

    if (isConfirmed) {
        fetch('/restart', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('all').style.display = 'none';
        })
        .catch(error => console.error('Error stopping generation:', error));
    }
});

document.getElementById('clearButton').addEventListener('click', function() {
    const isConfirmed = confirm('Are you sure you want to clear all images?');

    if (isConfirmed) {
        fetch('/clear', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            existingImages.clear();
            document.getElementById('images').innerHTML = '';
        })
        .catch(error => console.error('Error Clearing Images:', error));
    }
});

//TODO handle prompt example change

const promptSelectElement = document.getElementById('example_prompt');
const promptTextareaElement = document.getElementById('prompt');

// Add an event listener for the 'change' event on the select element
promptSelectElement.addEventListener('change', function() {
    promptTextareaElement.value = promptSelectElement.value; // Update textarea with the selected prompt
});

// Set the initial value of the textarea to the first option's value
promptTextareaElement.value = promptSelectElement.value;

//TODO handle model change

const modelSelectElement = document.getElementById('model');
const cfgInputElement = document.getElementById('cfg_scale');

// Add an event listener for the 'change' event on the select element
modelSelectElement.addEventListener('change', function() {
    cfgInputElement.value = modelSelectElement.options[modelSelectElement.selectedIndex].dataset.cfg || 7;
});

// Set the initial value of the textarea to the first option's value
promptTextareaElement.value = promptSelectElement.value;

//TODO handle pre dimension change

// Assuming the following elements exist in your HTML
const exampleSizeSelectElement = document.getElementById('example_size');
const widthInputElement = document.getElementById('width');
const heightInputElement = document.getElementById('height');

// Add an event listener for the 'change' event on the example_size select element
exampleSizeSelectElement.addEventListener('change', function() {
    const selectedOption = exampleSizeSelectElement.options[exampleSizeSelectElement.selectedIndex];

    if (selectedOption) {
        // Parse the selected value to get width and height
        const dimensions = selectedOption.value.split('x'); // Assuming the value format is "widthxheight"
        if (dimensions.length === 2) {
            widthInputElement.value = dimensions[0]; // Set width
            heightInputElement.value = dimensions[1]; // Set height
        }
    }
});

// Optionally, set the initial values based on the first option in the select
if (exampleSizeSelectElement.options.length > 0) {
    const initialDimensions = exampleSizeSelectElement.options[0].value.split('x');
    widthInputElement.value = initialDimensions[0];
    heightInputElement.value = initialDimensions[1];
}