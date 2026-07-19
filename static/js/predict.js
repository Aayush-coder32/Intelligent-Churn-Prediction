function buildDriverMarkup(drivers, type) {
    if (!drivers || !drivers.length) {
        return "<p class='text-secondary small mb-0'>No strong feature contributions were available for this prediction.</p>";
    }

    return `
        <div class="driver-list">
            ${drivers.map((driver) => `
                <div class="driver-item ${type}">
                    <strong>${driver.feature}</strong>
                    <div class="small text-secondary mt-1">${driver.direction}</div>
                    <div class="small fw-semibold mt-1">Impact: ${driver.impact}</div>
                </div>
            `).join("")}
        </div>
    `;
}

function renderResult(payload) {
    const resultContainer = document.getElementById("predictionResult");
    resultContainer.innerHTML = `
        <div class="section-head">
            <h3>Prediction Output</h3>
            <p>${payload.explanation.summary}</p>
        </div>
        <div class="d-flex align-items-center justify-content-between flex-wrap gap-3 mt-3">
            <div>
                <span class="prediction-pill">${payload.prediction}</span>
                <h2 class="mt-3 mb-1">${payload.probability}%</h2>
                <p class="text-secondary mb-0">Risk Level: <strong>${payload.risk_level}</strong></p>
            </div>
            <a href="${payload.report_url}" class="btn btn-outline-light">Download PDF</a>
        </div>
        <div class="mt-4">
            <div class="d-flex justify-content-between mb-2">
                <span>Confidence Meter</span>
                <span>${payload.probability}%</span>
            </div>
            <div class="risk-meter">
                <span style="width: ${payload.probability}%"></span>
            </div>
        </div>
        <div class="mt-4">
            <h4>Top Churn Drivers</h4>
            ${buildDriverMarkup(payload.explanation.positive_drivers, "positive")}
        </div>
        <div class="mt-4">
            <h4>Retention Signals</h4>
            ${buildDriverMarkup(payload.explanation.negative_drivers, "negative")}
        </div>
        <div class="mt-4">
            <h4>Recommendations</h4>
            <ul class="recommendation-list">
                ${payload.recommendations.map((item) => `<li>${item}</li>`).join("")}
            </ul>
        </div>
    `;
}

async function handlePredictionSubmit(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const formData = new FormData(form);
    const payload = Object.fromEntries(formData.entries());

    ChurnIQ.setLoadingState(true);
    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(payload),
        });
        const result = await ChurnIQ.parseApiResponse(response);

        renderResult(result);
        ChurnIQ.showToast("success", "Prediction generated", `Risk level: ${result.risk_level}`);
    } catch (error) {
        ChurnIQ.handleApiError(error, "Prediction failed");
    } finally {
        ChurnIQ.setLoadingState(false);
    }
}

document.addEventListener("DOMContentLoaded", () => {
    const predictionForm = document.getElementById("predictionForm");
    if (predictionForm) {
        predictionForm.addEventListener("submit", handlePredictionSubmit);
    }
});
