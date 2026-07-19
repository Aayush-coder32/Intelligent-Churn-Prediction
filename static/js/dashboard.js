const dashboardCharts = {};

const centerLabelPlugin = {
    id: "centerLabelPlugin",
    afterDatasetsDraw(chart, args, pluginOptions) {
        if (chart.config.type !== "doughnut" || !pluginOptions?.enabled) {
            return;
        }

        const dataset = chart.data.datasets?.[0];
        if (!dataset?.data?.length) {
            return;
        }

        const meta = chart.getDatasetMeta(0);
        const firstArc = meta?.data?.[0];
        if (!firstArc) {
            return;
        }

        const { ctx } = chart;
        const centerX = firstArc.x;
        const centerY = firstArc.y;
        const textColor = getCssVar("--text-primary");
        const subTextColor = getCssVar("--text-secondary");

        ctx.save();
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";

        ctx.fillStyle = subTextColor;
        ctx.font = "600 12px Manrope";
        ctx.fillText(pluginOptions.label || "Leading", centerX, centerY - 12);

        ctx.fillStyle = textColor;
        ctx.font = "700 22px Space Grotesk";
        ctx.fillText(pluginOptions.value || "", centerX, centerY + 10);
        ctx.restore();
    },
};

Chart.register(centerLabelPlugin);

function getCssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function destroyExistingChart(canvasId) {
    if (dashboardCharts[canvasId]) {
        dashboardCharts[canvasId].destroy();
    }
}

function createVerticalGradient(canvas, colors) {
    const context = canvas.getContext("2d");
    const gradient = context.createLinearGradient(0, 0, 0, canvas.height || 320);
    colors.forEach(([stop, color]) => gradient.addColorStop(stop, color));
    return gradient;
}

function sumValues(dataset) {
    return Object.values(dataset).reduce((total, value) => total + Number(value || 0), 0);
}

function findPeakEntry(dataset) {
    const entries = Object.entries(dataset);
    if (!entries.length) {
        return ["-", 0];
    }
    return entries.reduce((best, current) => (Number(current[1]) > Number(best[1]) ? current : best), entries[0]);
}

function formatPercent(value) {
    return `${Number(value || 0).toFixed(1)}%`;
}

function baseCartesianOptions() {
    return {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
            mode: "index",
            intersect: false,
        },
        plugins: {
            legend: {
                labels: {
                    color: getCssVar("--text-primary"),
                    usePointStyle: true,
                    pointStyle: "circle",
                    padding: 16,
                },
            },
            tooltip: {
                backgroundColor: "rgba(15, 23, 42, 0.92)",
                titleColor: "#f8fafc",
                bodyColor: "#e2e8f0",
                padding: 12,
                cornerRadius: 14,
                displayColors: true,
            },
        },
        scales: {
            x: {
                grid: {
                    display: false,
                },
                ticks: {
                    color: getCssVar("--text-secondary"),
                    font: {
                        size: 11,
                    },
                },
            },
            y: {
                beginAtZero: true,
                grid: {
                    color: "rgba(148, 163, 184, 0.12)",
                },
                ticks: {
                    color: getCssVar("--text-secondary"),
                    font: {
                        size: 11,
                    },
                },
            },
        },
    };
}

function renderMonthlyPredictionsChart(dataset) {
    const canvasId = "monthlyPredictionsChart";
    const canvas = document.getElementById(canvasId);
    if (!canvas) {
        return;
    }

    destroyExistingChart(canvasId);

    const labels = Object.keys(dataset);
    const values = Object.values(dataset).map((value) => Number(value || 0));
    const totalVolume = sumValues(dataset);
    const [peakLabel] = findPeakEntry(dataset);
    const activeWindows = values.filter((value) => value > 0).length;

    document.getElementById("monthlyVolumeTotal").textContent = totalVolume;
    document.getElementById("monthlyPeakPeriod").textContent = peakLabel;
    document.getElementById("monthlyActiveWindows").textContent = activeWindows;

    const areaGradient = createVerticalGradient(canvas, [
        [0, "rgba(15, 118, 110, 0.35)"],
        [0.65, "rgba(15, 118, 110, 0.14)"],
        [1, "rgba(15, 118, 110, 0.02)"],
    ]);

    dashboardCharts[canvasId] = new Chart(canvas, {
        type: "line",
        data: {
            labels,
            datasets: [
                {
                    label: "Prediction Volume",
                    data: values,
                    borderColor: "#0f766e",
                    backgroundColor: areaGradient,
                    fill: true,
                    borderWidth: 3,
                    tension: 0.38,
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    pointBackgroundColor: "#f8fafc",
                    pointBorderColor: "#0f766e",
                    pointBorderWidth: 2,
                },
            ],
        },
        options: {
            ...baseCartesianOptions(),
            plugins: {
                ...baseCartesianOptions().plugins,
                legend: {
                    display: false,
                },
                tooltip: {
                    ...baseCartesianOptions().plugins.tooltip,
                    callbacks: {
                        label(context) {
                            return ` ${context.dataset.label}: ${context.parsed.y}`;
                        },
                    },
                },
            },
        },
    });
}

function renderChurnTrendChart(churnDataset, volumeDataset) {
    const canvasId = "churnTrendChart";
    const canvas = document.getElementById(canvasId);
    if (!canvas) {
        return;
    }

    destroyExistingChart(canvasId);

    const labels = Object.keys(churnDataset);
    const churnCounts = labels.map((label) => Number(churnDataset[label] || 0));
    const volumeCounts = labels.map((label) => Number(volumeDataset[label] || 0));
    const churnRates = labels.map((label, index) => {
        const denominator = volumeCounts[index];
        return denominator > 0 ? Number(((churnCounts[index] / denominator) * 100).toFixed(1)) : 0;
    });

    const [, peakValue] = findPeakEntry(churnDataset);
    const averageRate = churnRates.length
        ? churnRates.reduce((total, value) => total + value, 0) / churnRates.length
        : 0;

    document.getElementById("churnSpikeValue").textContent = peakValue;
    document.getElementById("churnAverageRate").textContent = formatPercent(averageRate);

    const barGradient = createVerticalGradient(canvas, [
        [0, "rgba(249, 115, 22, 0.92)"],
        [1, "rgba(249, 115, 22, 0.22)"],
    ]);

    dashboardCharts[canvasId] = new Chart(canvas, {
        data: {
            labels,
            datasets: [
                {
                    type: "bar",
                    label: "Churn Alerts",
                    data: churnCounts,
                    backgroundColor: barGradient,
                    borderRadius: 12,
                    borderSkipped: false,
                    maxBarThickness: 42,
                    yAxisID: "y",
                },
                {
                    type: "line",
                    label: "Alert Rate %",
                    data: churnRates,
                    borderColor: "#0ea5e9",
                    backgroundColor: "#0ea5e9",
                    borderWidth: 2.5,
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    pointBackgroundColor: "#0ea5e9",
                    tension: 0.35,
                    yAxisID: "y1",
                },
            ],
        },
        options: {
            ...baseCartesianOptions(),
            plugins: {
                ...baseCartesianOptions().plugins,
                legend: {
                    ...baseCartesianOptions().plugins.legend,
                    display: true,
                },
            },
            scales: {
                ...baseCartesianOptions().scales,
                y: {
                    ...baseCartesianOptions().scales.y,
                    title: {
                        display: true,
                        text: "Churn Alerts",
                        color: getCssVar("--text-secondary"),
                    },
                },
                y1: {
                    beginAtZero: true,
                    position: "right",
                    grid: {
                        drawOnChartArea: false,
                    },
                    ticks: {
                        color: getCssVar("--text-secondary"),
                        callback(value) {
                            return `${value}%`;
                        },
                    },
                    title: {
                        display: true,
                        text: "Alert Rate",
                        color: getCssVar("--text-secondary"),
                    },
                },
            },
        },
    });
}

function renderContractLegend(dataset, palette) {
    const legendContainer = document.getElementById("contractLegend");
    if (!legendContainer) {
        return;
    }

    const total = sumValues(dataset);
    const entries = Object.entries(dataset).sort((left, right) => Number(right[1]) - Number(left[1]));

    legendContainer.innerHTML = entries.map(([label, value], index) => {
        const numericValue = Number(value || 0);
        const share = total > 0 ? (numericValue / total) * 100 : 0;
        return `
            <div class="contract-legend-item">
                <div class="contract-legend-head">
                    <span class="contract-dot" style="background:${palette[index % palette.length]}"></span>
                    <div>
                        <strong>${label}</strong>
                        <p>${numericValue.toLocaleString()} customers</p>
                    </div>
                    <span class="contract-share">${share.toFixed(1)}%</span>
                </div>
                <div class="contract-progress">
                    <span style="width:${share}%; background:${palette[index % palette.length]}"></span>
                </div>
            </div>
        `;
    }).join("");
}

function renderContractDistributionChart(dataset) {
    const canvasId = "contractChart";
    const canvas = document.getElementById(canvasId);
    if (!canvas) {
        return;
    }

    destroyExistingChart(canvasId);

    const labels = Object.keys(dataset);
    const values = Object.values(dataset).map((value) => Number(value || 0));
    const palette = ["#0f766e", "#f59e0b", "#0ea5e9", "#fb7185"];
    const [leadingLabel, leadingValue] = findPeakEntry(dataset);
    const total = sumValues(dataset);
    const leaderShare = total > 0 ? `${Math.round((Number(leadingValue) / total) * 100)}%` : "0%";

    renderContractLegend(dataset, palette);

    dashboardCharts[canvasId] = new Chart(canvas, {
        type: "doughnut",
        data: {
            labels,
            datasets: [
                {
                    data: values,
                    backgroundColor: palette,
                    borderColor: getCssVar("--bg-primary"),
                    borderWidth: 5,
                    hoverOffset: 10,
                    cutout: "68%",
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false,
                },
                tooltip: {
                    backgroundColor: "rgba(15, 23, 42, 0.92)",
                    titleColor: "#f8fafc",
                    bodyColor: "#e2e8f0",
                    padding: 12,
                    cornerRadius: 14,
                    callbacks: {
                        label(context) {
                            const value = Number(context.parsed || 0);
                            const share = total > 0 ? ((value / total) * 100).toFixed(1) : 0;
                            return ` ${context.label}: ${value.toLocaleString()} (${share}%)`;
                        },
                    },
                },
                centerLabelPlugin: {
                    enabled: true,
                    label: leadingLabel,
                    value: leaderShare,
                },
            },
        },
    });
}

function renderBasicChart(canvasId, chartType, dataset, color) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) {
        return;
    }

    destroyExistingChart(canvasId);

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
                    borderRadius: chartType === "bar" ? 12 : 0,
                    borderSkipped: false,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: chartType === "pie" || chartType === "doughnut",
                    labels: { color: getCssVar("--text-primary") },
                },
            },
            scales: chartType === "line" || chartType === "bar" ? {
                x: { ticks: { color: getCssVar("--text-secondary") }, grid: { display: false } },
                y: {
                    beginAtZero: true,
                    ticks: { color: getCssVar("--text-secondary") },
                    grid: { color: "rgba(148, 163, 184, 0.12)" },
                },
            } : {},
        },
    });
}

async function loadDashboard() {
    ChurnIQ.setLoadingState(true);
    try {
        const response = await fetch("/api/dashboard");
        const payload = await ChurnIQ.parseApiResponse(response);

        document.getElementById("cardTotalCustomers").textContent = payload.cards.total_customers ?? 0;
        document.getElementById("cardTotalPredictions").textContent = payload.cards.total_predictions ?? 0;
        document.getElementById("cardHighRisk").textContent = payload.cards.high_risk_customers ?? 0;
        document.getElementById("cardMonthlyCharges").textContent = `$${(payload.cards.avg_monthly_charges ?? 0).toFixed(2)}`;
        document.getElementById("cardChurnRate").textContent = `${payload.cards.churn_rate ?? 0}%`;

        renderMonthlyPredictionsChart(payload.charts.monthly_predictions);
        renderChurnTrendChart(payload.charts.churn_trend, payload.charts.monthly_predictions);
        renderContractDistributionChart(payload.charts.contract_distribution);
        renderBasicChart("genderChart", "pie", payload.charts.gender_distribution, ["#0ea5e9", "#fb7185", "#64748b"]);
        renderBasicChart("paymentChart", "bar", payload.charts.payment_method_distribution, "#0ea5e9");
        renderBasicChart("internetChart", "bar", payload.charts.internet_service_distribution, "#16a34a");
        renderBasicChart("chargesChart", "bar", payload.charts.monthly_charge_distribution, "#f59e0b");
        renderBasicChart("tenureChart", "bar", payload.charts.tenure_distribution, "#3b82f6");
    } catch (error) {
        ChurnIQ.handleApiError(error, "Dashboard failed to load");
    } finally {
        ChurnIQ.setLoadingState(false);
    }
}

document.addEventListener("DOMContentLoaded", loadDashboard);
