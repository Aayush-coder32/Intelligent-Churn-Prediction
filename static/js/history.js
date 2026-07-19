function buildHistoryRow(item) {
    return `
        <tr>
            <td>
                <strong>${item.customer_name}</strong>
            </td>
            <td>${item.prediction}</td>
            <td>${item.probability}%</td>
            <td>${item.risk_level}</td>
            <td>${item.created_at}</td>
            <td class="d-flex gap-2 flex-wrap">
                <a class="btn btn-sm btn-outline-light" href="${item.report_url}">PDF</a>
                <button class="btn btn-sm btn-outline-danger" data-delete-id="${item.id}" type="button">Delete</button>
            </td>
        </tr>
    `;
}

async function loadHistory(query = "") {
    const tableBody = document.getElementById("historyTableBody");
    ChurnNet.setLoadingState(true);
    try {
        const url = query ? `/api/history?q=${encodeURIComponent(query)}` : "/api/history";
        const response = await fetch(url);
        const payload = await ChurnNet.parseApiResponse(response);

        if (!payload.items.length) {
            tableBody.innerHTML = `
                <tr>
                    <td colspan="6" class="text-center text-secondary">No matching predictions found.</td>
                </tr>
            `;
            return;
        }

        tableBody.innerHTML = payload.items.map(buildHistoryRow).join("");
    } catch (error) {
        ChurnNet.handleApiError(error, "History failed to load");
    } finally {
        ChurnNet.setLoadingState(false);
    }
}

async function deletePrediction(predictionId) {
    const confirmation = await Swal.fire({
        icon: "warning",
        title: "Delete this prediction?",
        text: "This action removes the saved history entry.",
        showCancelButton: true,
        confirmButtonText: "Delete",
    });

    if (!confirmation.isConfirmed) {
        return;
    }

    try {
        const response = await fetch(`/api/prediction/${predictionId}`, {
            method: "DELETE",
        });
        const result = await ChurnNet.parseApiResponse(response);
        ChurnNet.showToast("success", result.message);
        loadHistory(document.getElementById("historySearch").value.trim());
    } catch (error) {
        ChurnNet.handleApiError(error, "Delete failed");
    }
}

document.addEventListener("DOMContentLoaded", () => {
    loadHistory();

    const searchInput = document.getElementById("historySearch");
    if (searchInput) {
        searchInput.addEventListener("input", (event) => {
            loadHistory(event.target.value.trim());
        });
    }

    const tableBody = document.getElementById("historyTableBody");
    if (tableBody) {
        tableBody.addEventListener("click", (event) => {
            const button = event.target.closest("[data-delete-id]");
            if (!button) {
                return;
            }
            deletePrediction(button.dataset.deleteId);
        });
    }
});
