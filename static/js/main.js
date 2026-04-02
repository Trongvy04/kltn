let csvData = [];
let currentIndex = 0;
let running = false;

const WARMUP_BARS = 80;
const DISPLAY_WINDOW = 30;
let normalDelay = 5000; // mặc định 5s

// ===== Data arrays =====
let labels = [];
let priceData = [];
let portfolioData = [];


// ===== Chart =====
const priceChart = new Chart(
    document.getElementById("priceChart"),
    {
        type: "line",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Close Price",
                    data: priceData,
                    borderColor: "blue",
                    pointRadius: 0,
                    tension: 0.2
                }
            ]
        }
    }
);

const portfolioChart = new Chart(
    document.getElementById("portfolioChart"),
    {
        type: "line",
        data: {
            labels: labels,
            datasets: [{
                label: "Portfolio Value",
                data: portfolioData,
                borderColor: "green",
                pointRadius: 0
            }]
        }
    }
);

// ===== CSV Reader =====
document.getElementById("csvFile").addEventListener("change", function(e) {

    const reader = new FileReader();

    reader.onload = function(event) {

        const rows = event.target.result.split("\n");
        csvData = [];

        for (let i = 1; i < rows.length; i++) {

            const cols = rows[i].split(",");
            if (cols.length < 6) continue;

            let date = cols[0];
            let o = parseFloat(cols[1]);
            let h = parseFloat(cols[2]);
            let l = parseFloat(cols[3]);
            let c = parseFloat(cols[4]);
            let v = parseFloat(cols[5]);

            if (!date || isNaN(o) || isNaN(h) || isNaN(l) || isNaN(c) || isNaN(v)) continue;

            csvData.push({
                date: date,
                open: o,
                high: h,
                low: l,
                close: c,
                volume: v
            });
        }

        console.log("CSV loaded:", csvData.length);
    };

    reader.readAsText(e.target.files[0]);
});

// ===== Simulation Control =====
function startSimulation() {

    if (running || csvData.length === 0) return;

    running = true;
    processNext();
}

function processNext() {

    if (!running || currentIndex >= csvData.length) {
        running = false;
        return;
    }

    sendStep(csvData[currentIndex]);
    currentIndex++;

    if (currentIndex < WARMUP_BARS) {
        processNext();
    } else {
        setTimeout(processNext, normalDelay);
    }
}

function pauseSimulation() {
    running = false;
}

function resetSimulation() {

    running = false;

    fetch("/reset", { method: "POST" });

    currentIndex = 0;

    labels.length = 0;
    priceData.length = 0;
    portfolioData.length = 0;

    document.getElementById("action").innerText = "-";
    document.getElementById("shares").innerText = "0.0000";
    document.getElementById("cash").innerText = "0.00";
    document.getElementById("portfolio").innerText = "0.00";

    priceChart.update();
    portfolioChart.update();
}

// ===== Send candle to BE =====
function sendStep(bar) {

    fetch("/step", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(bar)
    })
    .then(res => res.json())
    .then(data => {

        if (!data.ready) return;

        updateUI(data);
    });
}

// ===== Update UI =====
function updateUI(data) {

    document.getElementById("action").innerText = data.action_name;
    document.getElementById("shares").innerText = Number(data.shares).toFixed(4);
    document.getElementById("cash").innerText = Number(data.cash).toFixed(2);
    document.getElementById("portfolio").innerText = Number(data.portfolio_value).toFixed(2);

    if (labels.length >= DISPLAY_WINDOW) {
        labels.shift();
        priceData.shift();
        portfolioData.shift();
    }

    labels.push(data.date);
    priceData.push(data.close_price);
    portfolioData.push(data.portfolio_value);

    priceChart.update();
    portfolioChart.update();
}
// ===== Speed Control =====
const speedSlider = document.getElementById("speedSlider");
const speedValue = document.getElementById("speedValue");

if (speedSlider) {
    speedSlider.addEventListener("input", function () {
        normalDelay = parseInt(this.value);
        speedValue.innerText = (normalDelay / 1000).toFixed(1) + "s";
    });
}