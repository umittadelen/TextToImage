// Reusable function to load JSON and populate a select element
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