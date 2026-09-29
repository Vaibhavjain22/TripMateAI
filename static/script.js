let currentThreadId = localStorage.getItem("travel_thread_id") || null;
let latestAnswerMarkdown = "";

function setPrompt(text) {
    document.getElementById("userInput").value = text;
}

function setLoading(isLoading) {
    const sendBtn = document.getElementById("sendBtn");
    const btnText = document.getElementById("btnText");
    const btnLoader = document.getElementById("btnLoader");

    sendBtn.disabled = isLoading;

    if (isLoading) {
        btnText.classList.add("hidden");
        btnLoader.classList.remove("hidden");
    } else {
        btnText.classList.remove("hidden");
        btnLoader.classList.add("hidden");
    }
}

function showError(message) {
    const errorBox = document.getElementById("errorBox");

    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
}

function hideError() {
    const errorBox = document.getElementById("errorBox");

    errorBox.classList.add("hidden");
    errorBox.textContent = "";
}

function renderAgentBadges(selectedAgents) {
    const badgesContainer = document.getElementById("agentBadges");
    if (!badgesContainer) return;

    if (!selectedAgents || selectedAgents.length === 0) {
        badgesContainer.innerHTML = "";
        return;
    }

    const agentLabels = {
        "flight_agent": "✈️ Flight Agent",
        "hotel_agent": "🏨 Hotel Agent",
        "weather_agent": "☀️ Weather Agent",
        "budget_agent": "💰 Budget Analyst",
        "itinerary_agent": "🗺️ Itinerary Agent"
    };

    badgesContainer.innerHTML = selectedAgents
        .map(agent => `<span class="agent-chip">${agentLabels[agent] || agent}</span>`)
        .join("");
}

function toggleRevisionBox() {
    const revisionArea = document.getElementById("revisionArea");
    if (revisionArea) {
        revisionArea.classList.toggle("hidden");
    }
}

function showResult(data) {
    if (typeof data === "string") {
        data = { answer: data, thread_id: currentThreadId };
    }

    if (data.guardrail_allowed === false) {
        showError(data.guardrail_reason || "Request was blocked by safety guardrails.");
        document.getElementById("resultSection").classList.add("hidden");
        return;
    }

    latestAnswerMarkdown = data.answer || "";

    const resultSection = document.getElementById("resultSection");
    const resultBox = document.getElementById("resultBox");
    const threadInfo = document.getElementById("threadInfo");
    const resultTitle = document.getElementById("resultTitle");
    const approvalBox = document.getElementById("approvalBox");
    const approvalMsg = document.getElementById("approvalMsg");

    if (typeof marked !== "undefined") {
        resultBox.innerHTML = marked.parse(latestAnswerMarkdown);
    } else {
        resultBox.innerText = latestAnswerMarkdown;
    }

    threadInfo.textContent = `Thread ID: ${data.thread_id || currentThreadId}`;
    renderAgentBadges(data.selected_agents);

    // Handle Human-in-the-Loop review state
    if (data.requires_approval) {
        if (approvalBox) {
            approvalBox.classList.remove("hidden");
            if (data.approval_request && approvalMsg) {
                approvalMsg.textContent = data.approval_request;
            }
        }
        if (resultTitle) {
            resultTitle.textContent = "Draft Itinerary (Pending Review)";
        }
    } else {
        if (approvalBox) {
            approvalBox.classList.add("hidden");
        }
        if (resultTitle) {
            resultTitle.textContent = "Your Finalized Travel Plan";
        }
    }

    resultSection.classList.remove("hidden");
    resultSection.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
}

async function submitApproval(approved) {
    hideError();

    const feedbackInput = document.getElementById("feedbackInput");
    const feedback = feedbackInput ? feedbackInput.value.trim() : "";

    if (!approved && !feedback) {
        showError("Please enter your revision feedback before submitting changes.");
        return;
    }

    const btnText = approved ? document.getElementById("approveBtnText") : document.getElementById("revisionBtnText");
    const btnLoader = approved ? document.getElementById("approveLoader") : document.getElementById("revisionLoader");
    const approveBtn = document.getElementById("approveBtn");
    const revisionSubmitBtn = document.getElementById("revisionSubmitBtn");

    if (btnText && btnLoader) {
        btnText.classList.add("hidden");
        btnLoader.classList.remove("hidden");
    }
    if (approveBtn) approveBtn.disabled = true;
    if (revisionSubmitBtn) revisionSubmitBtn.disabled = true;

    try {
        const response = await fetch("/api/travel/approve", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                thread_id: currentThreadId,
                approved: approved,
                feedback: feedback
            })
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.error || "Approval request failed.");
        }

        if (data.thread_id) {
            currentThreadId = data.thread_id;
            localStorage.setItem("travel_thread_id", currentThreadId);
        }

        // Reset feedback input
        if (feedbackInput) {
            feedbackInput.value = "";
        }
        const revisionArea = document.getElementById("revisionArea");
        if (revisionArea) {
            revisionArea.classList.add("hidden");
        }

        showResult(data);

    } catch (error) {
        showError(error.message);
    } finally {
        if (btnText && btnLoader) {
            btnText.classList.remove("hidden");
            btnLoader.classList.add("hidden");
        }
        if (approveBtn) approveBtn.disabled = false;
        if (revisionSubmitBtn) revisionSubmitBtn.disabled = false;
    }
}

async function sendMessage() {
    hideError();

    const input = document.getElementById("userInput");
    const message = input.value.trim();

    if (!message) {
        showError("Please enter your travel request first.");
        return;
    }

    setLoading(true);

    try {
        const response = await fetch("/api/travel", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message: message,
                thread_id: null
            })
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.error || "Something went wrong.");
        }

        currentThreadId = data.thread_id;
        localStorage.setItem("travel_thread_id", currentThreadId);

        showResult(data);

    } catch (error) {
        showError(error.message);
    } finally {
        setLoading(false);
    }
}

function copyResult() {
    const resultBox = document.getElementById("resultBox");
    const text = resultBox.innerText;

    if (!text) {
        return;
    }

    navigator.clipboard.writeText(text)
        .then(() => {
            const copyBtn = document.querySelector(".copy-btn");
            const oldText = copyBtn.textContent;

            copyBtn.textContent = "Copied!";

            setTimeout(() => {
                copyBtn.textContent = oldText;
            }, 1400);
        })
        .catch(() => {
            showError("Could not copy result.");
        });
}

function downloadPDF() {
    const pdfContent = document.getElementById("pdfContent");

    if (!latestAnswerMarkdown || !pdfContent) {
        showError("No travel plan available to download.");
        return;
    }

    const downloadBtn = document.querySelector(".download-btn");
    const oldText = downloadBtn.textContent;

    downloadBtn.textContent = "Preparing PDF...";
    downloadBtn.disabled = true;

    const options = {
        margin: 0.5,
        filename: "ai-travel-plan.pdf",
        image: {
            type: "jpeg",
            quality: 0.98
        },
        html2canvas: {
            scale: 2,
            useCORS: true,
            backgroundColor: "#ffffff"
        },
        jsPDF: {
            unit: "in",
            format: "a4",
            orientation: "portrait"
        },
        pagebreak: {
            mode: ["avoid-all", "css", "legacy"]
        }
    };

    html2pdf()
        .set(options)
        .from(pdfContent)
        .save()
        .then(() => {
            downloadBtn.textContent = oldText;
            downloadBtn.disabled = false;
        })
        .catch(() => {
            downloadBtn.textContent = oldText;
            downloadBtn.disabled = false;
            showError("Could not download PDF.");
        });
}

document.addEventListener("keydown", function (event) {
    if (event.ctrlKey && event.key === "Enter") {
        sendMessage();
    }
});