function sendMessage() {
    const message = document.getElementById('user-input').value;
    if (!message) return;

    // Append user message to chat container
    document.getElementById('chat-container').innerHTML += `<div>User: ${message}</div>`;

    fetch('/get', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `msg=${encodeURIComponent(message)}`
    })
    .then(response => response.json())
    .then(data => {
        // Append bot response to chat container
        document.getElementById('chat-container').innerHTML += `<div>Bot: ${data.response}</div>`;
        document.getElementById('user-input').value = ''; // Clear input
    });
}

// Optionally, add an event listener for Enter key
document.getElementById('user-input').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        e.preventDefault();
        sendMessage();
    }
});