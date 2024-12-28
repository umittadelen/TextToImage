function loadTheme() {
    const savedData = localStorage.getItem('theme');
    if (savedData) {
        try {
            const theme = JSON.parse(savedData);

            // Apply the saved theme to the document
            document.documentElement.style.setProperty('--tone1', theme.tone_1);
            document.documentElement.style.setProperty('--tone2', theme.tone_2);
            document.documentElement.style.setProperty('--tone3', theme.tone_3);

            // Select the matching option in the dropdown
            try{
                const themeSelect = document.getElementById('theme_select');
                let matchedOption = Array.from(themeSelect.options).find(option => {
                    const decodedValue = JSON.parse(option.value.replace(/&quot;/g, '"')); // Decode HTML-encoded value
                    return JSON.stringify(decodedValue) === JSON.stringify(theme);
                });

                if (matchedOption) {
                    themeSelect.value = matchedOption.value; // Set the active option
                    themeSelect.dispatchEvent(new Event('change')); // Trigger 'change' event
                    console.log('Theme loaded and active option selected:', theme);
                } else {
                    console.warn('No matching option found in the dropdown for the saved theme.');
                }
            } catch (error) {
                console.error('Error applying theme:', error);
            }
        } catch (error) {
            console.error('Error applying theme:', error);
            localStorage.removeItem('theme'); // Remove invalid data to avoid future errors
        }
    } else {
        console.log('No theme found in localStorage.');
    }
}

window.onload = function() {
    loadTheme();
};