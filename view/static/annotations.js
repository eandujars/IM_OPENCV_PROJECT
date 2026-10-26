document.addEventListener('DOMContentLoaded', function () {
    const canvas = document.getElementById('annotation-canvas');
    if (!canvas) return;

    const context = canvas.getContext('2d');
    const color = document.getElementById('annotation-color');
    const size = document.getElementById('annotation-size');
    const eraser = document.getElementById('annotation-eraser');
    const undo = document.getElementById('annotation-undo');
    const clear = document.getElementById('clear-annotations');
    const modal = document.getElementById('annotation-modal');
    let history = [];
    let isEraser = false;

    function saveState() {
        history.push(context.getImageData(0, 0, canvas.width, canvas.height));
        if (history.length > 30) history.shift();
    }

    color.addEventListener('input', function () {
        isEraser = false;
        eraser.classList.remove('active');
        context.strokeStyle = color.value;
    });

    size.addEventListener('input', function () {
        context.lineWidth = Number(size.value);
    });

    eraser.addEventListener('click', function () {
        isEraser = !isEraser;
        eraser.classList.toggle('active', isEraser);
        context.strokeStyle = isEraser ? '#ffffff' : color.value;
    });

    undo.addEventListener('click', function () {
        const previous = history.pop();
        if (previous) context.putImageData(previous, 0, 0);
    });

    clear.addEventListener('click', function () {
        saveState();
        context.clearRect(0, 0, canvas.width, canvas.height);
    });

    canvas.addEventListener('pointerdown', saveState, true);

    document.getElementById('save-annotations').addEventListener('click', async function () {
        const body = new FormData();
        body.append('imagen', canvas.toDataURL('image/png'));
        const response = await fetch('/objetos/' + window.location.pathname.split('/')[2] + '/anotaciones', {
            method: 'POST',
            body
        });
        if (response.ok) {
            modal.hidden = true;
            alert('Anotaciones guardadas.');
        }
    });
});
