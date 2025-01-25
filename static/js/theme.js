async function loadTheme() {
    let savedData = "";

    try {
        const response = await fetch("/load_settings");
        const data = await response.json();

        // Fallback to default theme if no theme is found
        savedData = data.theme || {
            "tone_1": "240, 240, 240",
            "tone_2": "240, 218, 218",
            "tone_3": "240, 163, 163"
        };

        // Apply the saved theme to the document
        document.documentElement.style.setProperty('--tone1', savedData.tone_1);
        document.documentElement.style.setProperty('--tone2', savedData.tone_2);
        document.documentElement.style.setProperty('--tone3', savedData.tone_3);
    } catch (error) {
        console.error('Error loading settings:', error);

        // Use default theme in case of error
        const defaultTheme = {
            "tone_1": "240, 240, 240",
            "tone_2": "240, 218, 218",
            "tone_3": "240, 163, 163"
        };

        document.documentElement.style.setProperty('--tone1', defaultTheme.tone_1);
        document.documentElement.style.setProperty('--tone2', defaultTheme.tone_2);
        document.documentElement.style.setProperty('--tone3', defaultTheme.tone_3);
    }
}

window.onload = function () {
    loadTheme();
};
