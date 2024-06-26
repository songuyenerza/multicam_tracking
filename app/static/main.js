const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

function drawFrame(data) {
    const frames = data.frames;
    const overview = data.overview;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw the frames in a 2x4 grid
    frames.forEach((frameData, index) => {
        const img = new Image();
        img.src = 'data:image/jpeg;base64,' + frameData;
        img.onload = () => {
            const row = Math.floor(index / 4);
            const col = index % 4;
            const width = canvas.width / 4;
            const height = canvas.height / 3;
            ctx.drawImage(img, col * width, row * height, width, height);
        };
    });

    // Draw the overview image at the bottom
    const overviewImg = new Image();
    overviewImg.src = 'data:image/jpeg;base64,' + overview;
    overviewImg.onload = () => {
        const yOffset = canvas.height * 2 / 3;
        ctx.drawImage(overviewImg, 0, yOffset, canvas.width, canvas.height / 3);
    };
}

function fetchData() {
    const eventSource = new EventSource('/video_feed');
    eventSource.onmessage = function(event) {
        const data = JSON.parse(event.data);
        drawFrame(data);
    };
}

fetchData();
