const loadingOverlay = document.getElementById("loadingOverlay");

function setLoadingState(isLoading) {
    if (!loadingOverlay) {
        return;
    }
    loadingOverlay.classList.toggle("hidden", !isLoading);
}

function showToast(icon, title, text = "") {
    if (!window.Swal) {
        return;
    }
    Swal.fire({
        toast: true,
        position: "top-end",
        icon,
        title,
        text,
        showConfirmButton: false,
        timer: 2800,
        timerProgressBar: true,
    });
}

function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("churniq-theme", theme);
}

document.addEventListener("DOMContentLoaded", () => {
    const savedTheme = localStorage.getItem("churniq-theme") || "light";
    applyTheme(savedTheme);

    const themeToggle = document.getElementById("themeToggle");
    if (themeToggle) {
        themeToggle.addEventListener("click", () => {
            const currentTheme = document.documentElement.getAttribute("data-theme") || "light";
            applyTheme(currentTheme === "light" ? "dark" : "light");
        });
    }

    const flashPayload = document.getElementById("flashPayload");
    if (flashPayload) {
        const messages = JSON.parse(flashPayload.dataset.messages || "[]");
        messages.forEach(([category, message]) => {
            const iconMap = {
                success: "success",
                info: "info",
                warning: "warning",
                error: "error",
            };
            showToast(iconMap[category] || "info", message);
        });
    }
});

window.ChurnIQ = {
    setLoadingState,
    showToast,
};
