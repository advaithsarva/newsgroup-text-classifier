// Get the quick add button
const quickAddBtn = document.getElementById("quick-add-btn");

// Add an event listener for the button click
quickAddBtn.addEventListener("click", function() {
    alert("Quick Add feature coming soon!");
});

// Add more interactive features below
// Example: Toggle dark mode
let isDarkMode = false;

function toggleDarkMode() {
    if (isDarkMode) {
        document.body.style.backgroundColor = "#f4f4f4";
        document.querySelector('.header').style.backgroundColor = "#2c3e50";
        isDarkMode = false;
    } else {
        document.body.style.backgroundColor = "#333";
        document.querySelector('.header').style.backgroundColor = "#1a1a1a";
        isDarkMode = true;
    }
}

// You can bind a button or use the quick add button to toggle themes.
quickAddBtn.addEventListener("dblclick", toggleDarkMode);
