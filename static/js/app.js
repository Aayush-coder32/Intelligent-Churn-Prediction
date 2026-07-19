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

async function parseApiResponse(response) {
    const contentType = response.headers.get("content-type") || "";
    const isJson = contentType.includes("application/json");
    const payload = isJson ? await response.json() : null;

    if (!response.ok) {
        const message = payload?.error || `Request failed with status ${response.status}`;
        const error = new Error(message);
        error.status = response.status;
        error.payload = payload;
        throw error;
    }

    if (!isJson) {
        throw new Error("Server returned a non-JSON response.");
    }

    return payload;
}

function handleApiError(error, fallbackTitle) {
    if (error?.status === 401 && error?.payload?.login_url) {
        showToast("warning", "Session required", "Please log in to continue.");
        window.location.href = error.payload.login_url;
        return;
    }
    showToast("error", fallbackTitle, error.message);
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
    handleApiError,
    parseApiResponse,
    setLoadingState,
    showToast,
};
