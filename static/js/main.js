class CustomConfirm {
    constructor() {
        this.overlay = null;
        this.box = null;
        this.isActive = false; // Tracks if a dialog is currently active
        this.escKeyListener = null; // Store reference to the keydown listener
    }

    createConfirm(message, buttons, overlayReturnValue) {
        return new Promise((resolve) => {
            // Prevent creating multiple dialogs
            if (this.isActive) {
                console.warn("A confirm dialog is already active.");
                return;
            }
            this.isActive = true;

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
                    // Execute the button's value (function) and resolve
                    if (typeof buttonConfig.value === 'function') {
                        buttonConfig.value();
                    }
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

            // Add Esc key listener
            this.escKeyListener = (e) => {
                if (e.key === 'Escape') {
                    this.closeConfirm();
                    resolve(overlayReturnValue);
                }
            };
            document.addEventListener('keydown', this.escKeyListener);
        });
    }

    closeConfirm() {
        if (this.overlay) {
            this.overlay.classList.remove('show');
            this.box.classList.remove('show');
            this.overlay.addEventListener('transitionend', () => {
                if (this.overlay && this.overlay.parentNode) {
                    document.body.removeChild(this.overlay);
                    this.isActive = false; // Allow new dialogs to be created
                }
            });
        }

        // Remove Esc key listener
        if (this.escKeyListener) {
            document.removeEventListener('keydown', this.escKeyListener);
            this.escKeyListener = null;
        }
    }
}

const existingImages = new Map(); // Store existing images with their seeds as keys
let isGeneratingNewImages = false;
let pendingUpdates = false;
let isCleared = false;
const customConfirm = new CustomConfirm();

function loadJsonAndPopulateSelect(location, selectId, dataHandler) {
    fetch(location)
        .then(response => {
            if (!response.ok) {
                throw new Error(`Error loading ${location}`);
            }
            return response.json();
        })
        .then(data => dataHandler(data, document.getElementById(selectId)))
        .catch(error => console.error('Error:', error));
}

function populateModels(data, select) {
    Object.entries(data).forEach(([modelName, modelData]) => {
        const option = document.createElement('option');
        option.value = modelData.path;
        option.dataset.cfg = modelData.cfg || 7;
        option.dataset.clip_skip = modelData.clip_skip || 2;
        option.textContent = modelName;
        if (modelData.disabled) {
            option.disabled = true;
        }
        select.appendChild(option);
    });
    select.selectedIndex = 0;
    changeModelvalues();
}

function populateSchedulers(data, select) {
    data.schedulers.forEach(scheduler => {
        const option = document.createElement('option');
        option.value = scheduler;
        option.textContent = scheduler;
        select.appendChild(option);
    });
}

loadJsonAndPopulateSelect('/static/json/models.json', 'model_select', populateModels);
loadJsonAndPopulateSelect('/static/json/schedulers.json', 'scheduler_select', populateSchedulers);

loadFormData();

function submitButtonOnClick(event) {
    event.preventDefault();

    // Clear existing images for new generation
    isGeneratingNewImages = true;
    existingImages.clear();
    document.getElementById('images').innerHTML = '';
    saveFormData();
    const formElement = document.getElementById('generateForm');

    // Prepare data for the server
    const formData = new FormData(formElement);
    console.log('Form data:', formData);
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
}

setInterval(() => {
    if (document.visibilityState === 'visible' && !pendingUpdates && !isCleared) {
        pendingUpdates = true; // Prevent overlapping updates

        fetch('/status', { cache: 'no-store' })
            .then((response) => response.json())
            .then((data) => {

                processImageUpdates(data.images);

                if (data.images.length < existingImages.size) {
                    existingImages.clear();
                    document.getElementById('images').innerHTML = '';
                }
            })
            .catch((error) => {
                console.error('Error fetching status:', error);
            })
            .finally(() => {
                pendingUpdates = false;
                isCleared = false;
            });
    }
}, 2500);

document.addEventListener('contextmenu', function (event) {
    // Check if the target element is a textarea or input
    if (event.target.tagName === 'TEXTAREA' || event.target.tagName === 'INPUT') {
        // Allow the default context menu for textareas and inputs
        return;
    }

    // Prevent the default context menu for other elements
    event.preventDefault();

    // Create the custom context menu
    customConfirm.createConfirm('Quick Actions', [
        { text: 'Generate Images', value: () => submitButtonOnClick(event) },
        { text: 'Clear Images', value: () => clearButtonOnClick(event) },
        { text: 'Stop Generation', value: () => stopButtonOnClick(event) },
        { text: 'Get Metadata', value: () => window.open(`metadata`) }
    ], true);
});

function processImageUpdates(images) {
    const imagesDiv = document.getElementById('images');

    images.forEach((imgData) => {
        const key = imgData.seed;

        if (existingImages.has(key)) {
            const existingImg = existingImages.get(key).querySelector('img');
            if (existingImg.src !== imgData.img) {
                existingImg.src = imgData.img;
            }
        } else {
            const wrapper = document.createElement('div');
            wrapper.className = 'image-wrapper';

            const img = document.createElement('img');
            img.src = imgData.img;
            img.loading = "lazy";
            img.onclick = () => openLink("image/" + imgData.img.split('/').pop());

            wrapper.appendChild(img);
            imagesDiv.appendChild(wrapper);
            existingImages.set(key, wrapper);
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

function stopButtonOnClick() {
    fetch('/stop', {
        method: 'POST'
    })
    .catch(error => console.error('Error stopping generation:', error));
};

async function restartButtonOnClick() {
    const isConfirmed = await customConfirm.createConfirm(
        'Are you sure you want to restart the server?\nIt will reset all variables and has a chance to fail restarting.',
        [
            { text: 'Restart', value: true },
            { text: 'Cancel', value: false }
        ],
        false
    );

    if (isConfirmed) {
        isCleared = true;
        fetch('/restart', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('all').style.display = 'none';
        })
        .catch(error => console.error('Error stopping generation:', error));
    }
};

async function resetFormButtonOnClick() {
    const isConfirmed = await customConfirm.createConfirm('Are you sure you want to reset the form cache?<br>this cannot be undone!',
        [
            { text: 'Reset', value: true },
            { text: 'Cancel', value: false }
        ],
        false
    );
    
    if (isConfirmed) {
        fetch('/reset_form_data')
        .then(response => response.json())
        .then(data => {
            console.log('Form cache has been reset:', data);
            location.reload();
        })
        .catch(error => {
            console.error('Error resetting form cache:', error);
        });

        console.log('Form cache has been reset.');
    }
};

function loadFormData() {
    fetch('/load_form_data')
    .then(response => response.json())
    .then(data => {
        savedData = data;
        const form = document.getElementById('generateForm');
        for (const [key, value] of Object.entries(data)) {
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
    })
}

function saveFormData() {
    const formDataToSave = {};
    const formElement = document.getElementById("generateForm"); // Explicitly get the form element
    Array.from(formElement.elements).forEach(field => {
        if (field.name && ['TEXTAREA', 'SELECT', 'INPUT'].includes(field.tagName)) {
            formDataToSave[field.name] = field.value;
        }
    });
    fetch('/save_form_data', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(formDataToSave)
    })
    .then(response => response.json())
    .then(data => {
        console.log('Form data saved:', data);
    })
    .catch(error => {
        console.error('Error saving form data:', error);
    });
}

async function clearButtonOnClick() {
    const isConfirmed = await customConfirm.createConfirm(
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
};

function changeModelvalues() {
    const modelSelectElement = document.getElementById('model_select');
    const cfgInputElement = document.getElementById('cfg_scale_input');
    const clipSkipInputElement = document.getElementById('clip_skip_input');

    const selectedOption = modelSelectElement.options[modelSelectElement.selectedIndex];
    cfgInputElement.value = selectedOption.dataset.cfg || 7;
    clipSkipInputElement.value = selectedOption.dataset.clip_skip || 2;
}