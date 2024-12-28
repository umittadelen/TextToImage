class CustomConfirm {
    constructor() {
        this.overlay = null;
        this.box = null;
    }

    createConfirm(message, buttons, overlayReturnValue) {
        return new Promise((resolve) => {
            // Create overlay
            this.overlay = document.createElement('div');
            this.overlay.className = 'custom-confirm-overlay';

            // Create confirm box
            this.box = document.createElement('div');
            this.box.className = 'custom-confirm-box';

            // Add message
            const msg = document.createElement('p');
            message = message.replace(/\n/g, '<br>');
            msg.innerHTML = message;
            this.box.appendChild(msg);

            // Add button container
            const buttonContainer = document.createElement('div');
            buttonContainer.className = 'button-container';

            // Add buttons to the button container
            buttons.forEach((buttonConfig) => {
                const button = document.createElement('button');
                button.textContent = buttonConfig.text;
                button.addEventListener('click', () => {
                    this.closeConfirm();
                    resolve(buttonConfig.value);
                });
                buttonContainer.appendChild(button);
            });

            // Append button container to the box
            this.box.appendChild(buttonContainer);

            // Append the box to the overlay and the overlay to the document body
            this.overlay.appendChild(this.box);
            document.body.appendChild(this.overlay);

            // Force reflow to ensure the transition is applied
            window.getComputedStyle(this.overlay).opacity;

            // Add the show class to trigger the transition
            this.overlay.classList.add('show');
            this.box.classList.add('show');

            // Add overlay click listener
            this.overlay.addEventListener('click', (e) => {
                // Prevent click events from propagating when clicking the confirm box itself
                if (e.target === this.overlay) {
                    this.closeConfirm();
                    resolve(overlayReturnValue);
                }
            });
        });
    }

    closeConfirm() {
        if (this.overlay) {
            this.overlay.classList.remove('show');
            this.box.classList.remove('show');
            this.overlay.addEventListener('transitionend', () => {
                if (this.overlay && this.overlay.parentNode) {
                    document.body.removeChild(this.overlay);
                }
            });
        }
    }
}

const existingImages = new Map(); // Store existing images with their seeds as keys
let isGeneratingNewImages = false;
let pendingUpdates = false;
const customConfirm = new CustomConfirm();

document.getElementById('generateForm').addEventListener('submit', function (event) {
    event.preventDefault();

    // Save form data to localStorage
    const formDataToSave = {};
    Array.from(this.elements).forEach(field => {
        if (field.name && ['TEXTAREA', 'SELECT', 'INPUT'].includes(field.tagName)) {
            formDataToSave[field.name] = field.value;
        }
    });
    localStorage.setItem('formData', JSON.stringify(formDataToSave));
    localStorage.setItem('formExpiry', Date.now() + 7 * 24 * 60 * 60 * 1000); // 1 week expiry

    // Clear existing images for new generation
    isGeneratingNewImages = true;
    existingImages.clear();
    document.getElementById('images').innerHTML = '';

    // Prepare data for the server
    const formData = new FormData(this);
    fetch('/generate', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            isGeneratingNewImages = false;
            console.log('Server response:', data);
        })
        .catch(error => {
            console.error('Error:', error);
            isGeneratingNewImages = false;
        });
});

function loadFormData() {
    const savedData = JSON.parse(localStorage.getItem('formData'));
    const expiryTime = localStorage.getItem('formExpiry');

    if (savedData && expiryTime && Date.now() < expiryTime) {
        const form = document.getElementById('generateForm');
        for (const [key, value] of Object.entries(savedData)) {
            const field = form.elements[key];
            if (field && ['TEXTAREA', 'SELECT', 'INPUT'].includes(field.tagName)) {
                if (field.tagName === 'SELECT') {
                    // Set the selected option for <select>
                    Array.from(field.options).forEach(option => {
                        option.selected = option.value === value;
                    });
                } else {
                    // Set the value for <textarea> and <input>
                    field.value = value;
                }
            }
        }
    } else {
        // Clear expired data
        localStorage.removeItem('formData');
        localStorage.removeItem('formExpiry');
    }
}

document.getElementById('resetCacheButton').addEventListener('click', function () {
    // Remove form data and expiry from localStorage
    localStorage.removeItem('formData');
    localStorage.removeItem('formExpiry');
    
    // Refresh the page
    location.reload(); // This reloads the page, effectively resetting the form and clearing cache

    console.log('Form cache has been reset.');
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

                updateImageScales();

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
}, 2500);

function updateImageScales() {
    const images = document.querySelectorAll('#images img');
    const value = Number(document.getElementById('img_display_input').value); // Convert value to a number
    images.forEach(img => {
        img.style.width = `${100 / value - 4}vw`;
        console.log(100 / value - 2);
    });
}

document.getElementById('img_display_input').addEventListener('change', updateImageScales);

function updateProgressBars(data) {
    const progressText = document.getElementById('progress');
    const statusDiv = document.getElementById('status');
    const dynamicProgressBar = document.getElementById('dynamic-progress-bar');
    const alldynamicProgressBar = document.getElementById('all-dynamic-progress-bar');

    // Update progress value smoothly
    if (Number.isInteger(data.imgprogress)) {
        dynamicProgressBar.style.width = `calc(${data.imgprogress}%)`;
        progressText.innerHTML = `Progress: ${data.imgprogress}% Remaining: ${data.remainingimages}`;
        statusDiv.style.display = 'block';
    }
    else if (data.imgprogress === 'Done' || data.imgprogress === 'Generation Complete') {
        statusDiv.style.display = 'none';
    }
    else if (data.imgprogress === '') {
        statusDiv.style.display = 'none';
    }
    else if (data.imgprogress === 'Starting') {
        statusDiv.style.display = 'block';
    }
    else {
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

function getSizeSuffix() {
    const connection = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
    const sizeMap = {
        'slow-2g': '?size=mini',
        '2g': '?size=mini',
        '3g': '?size=small',
        '4g': '?size=medium'
    };
    const typesWithSuffix = ['cellular', 'wimax', 'bluetooth', 'other', 'unknown', 'none'];

    if (connection) {
        if (connection.type === 'cellular' && connection.effectiveType === '4g') {
            return '?size=small';
        }
        if (typesWithSuffix.includes(connection.type)) {
            return sizeMap[connection.effectiveType] || '';
        }
    }
    return '';
}

function updateImageSizes() {
    existingImages.forEach((wrapper) => {
        const img = wrapper.querySelector('img');
        const imgData = img.src.split('?')[0];
        const sizeSuffix = getSizeSuffix();
        img.src = `${imgData}${sizeSuffix}`;
    });
}

if (navigator.connection || navigator.mozConnection || navigator.webkitConnection) {
    (navigator.connection || navigator.mozConnection || navigator.webkitConnection).addEventListener('change', updateImageSizes);
}

function processImageUpdates(images) {
    const imagesDiv = document.getElementById('images');

    images.forEach((imgData) => {
        const key = imgData.seed;
        const sizeSuffix = getSizeSuffix();

        if (existingImages.has(key)) {
            const existingImg = existingImages.get(key).querySelector('img');
            if (existingImg.src !== imgData.img + sizeSuffix) {
                existingImg.src = imgData.img + sizeSuffix;
            }
        } else {
            const wrapper = document.createElement('div');
            wrapper.className = 'image-wrapper';

            const img = document.createElement('img');
            img.src = imgData.img + sizeSuffix;
            img.loading = "lazy";
            img.onclick = () => openLink("image/" + imgData.img.split('/').pop());

            wrapper.appendChild(img);
            imagesDiv.appendChild(wrapper);
            existingImages.set(key, wrapper);
            updateImageSizes();
        }
    });
}

function openLink(link) {
    window.open(link.split('?')[0]);
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

document.getElementById('restartButton').addEventListener('click', async function() {
    const isConfirmed = await customConfirm.createConfirm(
        'Are you sure you want to restart the server?\nIt will reset all variables and has a chance to fail restarting.',
        [
            { text: 'Restart', value: true },
            { text: 'Cancel', value: false }
        ],
        false
    );

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
    const isConfirmed = customConfirm.createConfirm(
        'Are you sure you want to clear all images?',
        [
            { text: 'Clear', value: true },
            { text: 'Cancel', value: false }
        ],
        false
    );

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

document.getElementById('theme_select').addEventListener('change', function () {
    try {
        const theme = JSON.parse(this.value);

        // Set CSS variables
        document.documentElement.style.setProperty('--tone1', theme.tone_1);
        document.documentElement.style.setProperty('--tone2', theme.tone_2);
        document.documentElement.style.setProperty('--tone3', theme.tone_3);

        // Save the theme to localStorage
        localStorage.setItem('theme', JSON.stringify(theme));
    } catch (error) {
        console.error('Error applying theme:', error);
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
const modelTypeInputElement = document.getElementById('model_type');

// Add an event listener for the 'change' event on the select element
modelSelectElement.addEventListener('change', function() {
    cfgInputElement.value = modelSelectElement.options[modelSelectElement.selectedIndex].dataset.cfg || 7;
    modelTypeInputElement.value = modelSelectElement.options[modelSelectElement.selectedIndex].dataset.type || "SDXL";
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