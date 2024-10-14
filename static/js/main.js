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
    fetch('/status')
        .then(response => response.json())
        .then(data => {
            document.getElementById('all').style.display = 'flex';

            const imagesDiv = document.getElementById('images');
            const progressText = document.getElementById('progress');
            const dynamicProgressBar = document.getElementById('dynamic-progress-bar');
            const alldynamicProgressBar = document.getElementById('all-dynamic-progress-bar');

            // Update progress value smoothly
            if (Number.isInteger(data.imgprogress)) {
                dynamicProgressBar.style.width = `${data.imgprogress}%`;
                alldynamicProgressBar.style.width = `${data.allPercentage}%`;
                progressText.innerHTML = `Progress: ${data.imgprogress}%`;
            } else {
                dynamicProgressBar.style.width = `0%`;
                alldynamicProgressBar.style.width = `0%`;
                progressText.innerHTML = `Progress: ${data.imgprogress}`;
            }

            // Only process images if not currently generating new ones
            if (!isGeneratingNewImages && data.images.length > 0) {
                data.images.forEach(imgData => {
                    const key = imgData.seed; // Assuming seed is the unique identifier for the image

                    if (existingImages.has(key)) {
                        // Update existing image
                        const existingWrapper = existingImages.get(key);
                        const existingImg = existingWrapper.querySelector('img');
                        existingImg.src = imgData.img;

                        // Update the blur effect based on the toggle
                        updateBlurEffect(existingImg, imgData.sensitive);
                    } else {
                        // Create new image element if it doesn't exist
                        const wrapper = document.createElement('div');
                        wrapper.className = 'image-wrapper';

                        const img = document.createElement('img');
                        img.src = imgData.img;
                        img.loading = "lazy";

                        // Create sensitive text if necessary
                        if (imgData.sensitive) {
                            const sensitiveText = document.createElement('p');
                            sensitiveText.classList.add('centered');
                            sensitiveText.innerHTML = imgData.sensitive;
                            wrapper.appendChild(sensitiveText);
                        }

                        // Apply initial blur effect
                        updateBlurEffect(img, imgData.sensitive);

                        //img.onclick = () => {location.href = imgData.img;};
                        img.onclick = () => {
                            openLink(imgData.img);
                        };

                        wrapper.appendChild(img); // Add image to wrapper
                        imagesDiv.appendChild(wrapper); // Add wrapper to imagesDiv

                        existingImages.set(key, wrapper); // Store the wrapper in the map
                    }
                });
            }
        })
        .catch(error => console.error('Error fetching status:', error));
}, 1000); // Check every 1 second

document.getElementById('stopButton').addEventListener('click', function() {
    fetch('/stop', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        console.log(data.status);
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
    fetch('/clear', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        existingImages.clear();
        document.getElementById('images').innerHTML = '';
    })
    .catch(error => console.error('Error Clearing Images:', error));
});

function updateBlurEffect(img, isSensitive) {
    if (isSensitive && !sensitiveToggle.checked) {
        img.classList.add('blurred'); // Add blur class if sensitive and toggle is off
    } else {
        img.classList.remove('blurred'); // Remove blur class if toggle is on or not sensitive
    }
}

function openLink(link) {
    window.location.href = link;
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

const selectElement = document.getElementById('example_prompt');
const SwitchTextareaElement = document.getElementById('prompt');

// Add an event listener for the 'change' event on the select element
selectElement.addEventListener('change', function() {
    SwitchTextareaElement.value = selectElement.value; // Update textarea with the selected prompt
});

// Set the initial value of the textarea to the first option's value
SwitchTextareaElement.value = selectElement.value;