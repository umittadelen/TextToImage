const existingImages = new Map(); // Store existing images with their seeds as keys
const sensitiveToggle = document.getElementById('sensitive-toggle');

let isGeneratingNewImages = false;

document.getElementById('generateForm').addEventListener('submit', function(event) {
    event.preventDefault();

    // Set the flag to true when a new generation starts
    isGeneratingNewImages = true;

    // Clear existing images for new generation
    existingImages.clear();
    document.getElementById('images').innerHTML = '';

    if (document.getElementById('model').value === "custom") {
        document.getElementById('model').value = document.getElementById('custom-model').value;
    }

    const formData = new FormData(this);
    fetch('/generate', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.seed) {
            globalSeed = data.seed; // Store the seed globally if necessary
        }

        // Set the flag to false when the generation request is acknowledged
        isGeneratingNewImages = false;
    })
    .catch(error => {
        console.error('Error:', error);
        isGeneratingNewImages = false; // Reset the flag in case of an error
    });
});

setInterval(() => {
    // Check if the document is visible before fetching the status
    if (document.visibilityState === 'visible') {
        fetch('/status', { cache: 'no-store' })
            .then(response => response.json())
            .then(data => {
                document.getElementById('all').style.display = 'flex';

                const imagesDiv = document.getElementById('images');
                const progressText = document.getElementById('progress');
                const dynamicProgressBar = document.getElementById('dynamic-progress-bar');
                const alldynamicProgressBar = document.getElementById('all-dynamic-progress-bar');

                // Update progress value smoothly
                if (Number.isInteger(data.imgprogress)) {
                    dynamicProgressBar.style.width = `calc(${data.imgprogress}% - 2px)`;
                    progressText.innerHTML = `Progress: ${data.imgprogress}% Remaining: ${data.remainingimages}`;
                } else {
                    dynamicProgressBar.style.width = `0%`;
                    alldynamicProgressBar.style.width = `0%`;
                    progressText.innerHTML = `Progress: ${data.imgprogress}`;
                }

                if (Number.isInteger(data.allpercentage)) {
                    alldynamicProgressBar.style.width = `calc(${data.allpercentage}% - 2px)`;
                } else {
                    alldynamicProgressBar.style.width = `0%`;
                }

                // Only process images if not currently generating new ones
                if (!isGeneratingNewImages && data.images.length > 0) {
                    data.images.forEach(imgData => {
                        const key = imgData.seed; // Assuming seed is the unique identifier for the image

                        if (existingImages.has(key)) {
                            // Update existing image
                            const existingWrapper = existingImages.get(key);
                            const existingImg = existingWrapper.querySelector('img');
                            existingImg.src = imgData.img + "?size=small";

                            // Update the blur effect based on the toggle
                            updateBlurEffect(existingImg, imgData.sensitive);
                        } else {
                            // Create new image element if it doesn't exist
                            const wrapper = document.createElement('div');
                            wrapper.className = 'image-wrapper';

                            const img = document.createElement('img');
                            img.src = imgData.img + "?size=small";
                            img.loading = "lazy";

                            // Create sensitive text if necessary
                            if (imgData.sensitive) {
                                const sensitiveText = document.createElement('p');
                                sensitiveText.classList.add('centered');
                                sensitiveText.innerHTML = "SENSITIVE";
                                wrapper.appendChild(sensitiveText);
                            }

                            // Apply initial blur effect
                            updateBlurEffect(img, imgData.sensitive);

                            img.onclick = () => {
                                openLink("image/" + imgData.img.split('/').pop()); // Send to "image/" + filename
                            };

                            wrapper.appendChild(img); // Add image to wrapper
                            imagesDiv.appendChild(wrapper); // Add wrapper to imagesDiv

                            existingImages.set(key, wrapper); // Store the wrapper in the map
                        }
                    });
                }
            })
            .catch(error => {
                console.error('Error fetching status:', error);
                document.getElementById('all').style.display = 'none';
            });
    }
}, 2000); // Check every 2 seconds

document.addEventListener('visibilitychange', function() {
    const state = document.visibilityState === 'visible' ? 'Vis' : 
                 document.visibilityState === 'hidden' ? 'Hid' : 
                 document.visibilityState === 'prerender' ? 'Pre' : 'Unk';
    document.title = `Image Generator (${state})`;
});


document.getElementById('stopButton').addEventListener('click', function() {
    fetch('/stop', {
        method: 'POST'
    })
    .catch(error => console.error('Error stopping generation:', error));
});

document.getElementById('restartButton').addEventListener('click', function() {
    fetch('/restart', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('all').style.display = 'none';
    })
    .catch(error => console.error('Error stopping generation:', error));
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

function updateBlurEffect(img, isSensitive) {
    if (isSensitive && !sensitiveToggle.checked) {
        img.classList.add('blurred'); // Add blur class if sensitive and toggle is off
    } else {
        img.classList.remove('blurred'); // Remove blur class if toggle is on or not sensitive
    }
}

function openLink(link) {
    window.location.href = link.replace("?size=small", "");
}

sensitiveToggle.addEventListener('change', () => {
    if (sensitiveToggle.checked) {
        // Ask for confirmation if the user is 18+
        const isAdult = confirm('Are you 18 years or older?');

        if (!isAdult) {
            // If user clicks "No", uncheck the toggle switch
            sensitiveToggle.checked = false;
            return; // Exit the function, no further action needed
        }
    }

    // Apply the blur effect based on the user's choice
    existingImages.forEach((wrapper) => {
        const img = wrapper.querySelector('img');
        const isSensitive = wrapper.querySelector('.centered') !== null; // Check if sensitive text exists
        updateBlurEffect(img, isSensitive);
    });
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

function toggleCustomModelInput() {
    const dropdown = document.getElementById('model');
    const customInput = document.getElementById('custom-model');
    if (dropdown.value === 'custom') {
        customInput.style.display = 'block';
        customInput.required = true; // Make input field required when visible
    } else {
        customInput.style.display = 'none';
        customInput.required = false;
    }
}