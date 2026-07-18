const dashboardCharts = {};

function renderChart(canvasId, chartType, dataset, color) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) {
        return;
    }

    if (dashboardCharts[canvasId]) {
        dashboardCharts[canvasId].destroy();
    }

    dashboardCharts[canvasId] = new Chart(canvas, {
        type: chartType,
        data: {
            labels: Object.keys(dataset),
            datasets: [
                {
                    data: Object.values(dataset),
                    label: canvasId,
                    backgroundColor: color,
                    borderColor: color,
                    borderWidth: 2,
                    tension: 0.35,
                    fill: chartType === "line",
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: chartType === "pie" || chartType === "doughnut",
                    labels: { color: getComputedStyle(document.documentElement).getPropertyValue("--text-primary") },
                },
            },
            scales: chartType === "line" || chartType === "bar" ? {
                x: { ticks: { color: getComputedStyle(document.documentElement).getPropertyValue("--text-secondary") } },
                y: {
                    beginAtZero: true,
                    ticks: { color: getComputedStyle(document.documentElement).getPropertyValue("--text-secondary") },
                },
            } : {},
        },
    });
}

async function loadDashboard() {
    ChurnIQ.setLoadingState(true);
    try {
        const response = await fetch("/api/dashboard");
        const payload = await response.json();

        document.getElementById("cardTotalCustomers").textContent = payload.cards.total_customers ?? 0;
        document.getElementById("cardTotalPredictions").textContent = payload.cards.total_predictions ?? 0;
        document.getElementById("cardHighRisk").textContent = payload.cards.high_risk_customers ?? 0;
        document.getElementById("cardMonthlyCharges").textContent = `$${(payload.cards.avg_monthly_charges ?? 0).toFixed(2)}`;
        document.getElementById("cardChurnRate").textContent = `${payload.cards.churn_rate ?? 0}%`;

        renderChart("monthlyPredictionsChart", "line", payload.charts.monthly_predictions, "#0f766e");
        renderChart("churnTrendChart", "bar", payload.charts.churn_trend, "#f97316");
        renderChart("contractChart", "doughnut", payload.charts.contract_distribution, ["#0f766e", "#f59e0b", "#0ea5e9"]);
        renderChart("genderChart", "pie", payload.charts.gender_distribution, ["#0ea5e9", "#fb7185", "#64748b"]);
        renderChart("paymentChart", "bar", payload.charts.payment_method_distribution, "#0ea5e9");
        renderChart("internetChart", "bar", payload.charts.internet_service_distribution, "#16a34a");
        renderChart("chargesChart", "bar", payload.charts.monthly_charge_distribution, "#f59e0b");
        renderChart("tenureChart", "bar", payload.charts.tenure_distribution, "#3b82f6");
    } catch (error) {
        ChurnIQ.showToast("error", "Dashboard failed to load", error.message);
    } finally {
        ChurnIQ.setLoadingState(false);
    }
}

document.addEventListener("DOMContentLoaded", loadDashboard);
