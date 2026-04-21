document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const tweetInput = document.getElementById('tweetInput');
    const resultDiv = document.getElementById('result');
    const sentimentBadge = document.getElementById('sentimentBadge');
    const probText = document.getElementById('probText');
    const probFill = document.getElementById('probFill');
    const cleanedTextP = document.getElementById('cleanedText');
    const loader = document.getElementById('loader');
    const btnText = document.querySelector('.btn-text');

    const botBadge = document.getElementById('botBadge');
    const botProbText = document.getElementById('botProbText');
    const botProbFill = document.getElementById('botProbFill');

    analyzeBtn.addEventListener('click', async () => {
        const text = tweetInput.value.trim();
        if (!text) return;

        // UI State: Loading
        analyzeBtn.disabled = true;
        loader.style.display = 'block';
        btnText.style.opacity = '0';
        resultDiv.style.display = 'none';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text })
            });

            const data = await response.json();

            if (data.error) {
                alert(data.error);
                return;
            }

            // Update UI with results
            resultDiv.style.display = 'grid'; // Use grid for side-by-side
            
            // 1. Sentiment Result
            sentimentBadge.textContent = data.sentiment;
            sentimentBadge.className = 'sentiment-badge ' + 
                (data.sentiment === 'Positive' ? 'sentiment-positive' : 'sentiment-negative');
            
            const sentConfidence = (data.probability > 0.5 ? data.probability : 1 - data.probability) * 100;
            probText.textContent = `Confidence: ${sentConfidence.toFixed(1)}%`;
            probFill.style.width = `${sentConfidence}%`;
            probFill.style.backgroundColor = data.sentiment === 'Positive' ? '#10b981' : '#ef4444';

            // 2. Bot Result
            botBadge.textContent = data.bot_label;
            botBadge.className = 'sentiment-badge ' + 
                (data.is_bot ? 'sentiment-negative' : 'sentiment-positive'); // Red for Bot, Green for Human
            
            const botConfidence = (data.bot_probability > 0.5 ? data.bot_probability : 1 - data.bot_probability) * 100;
            botProbText.textContent = `Confidence: ${botConfidence.toFixed(1)}%`;
            botProbFill.style.width = `${botConfidence}%`;
            botProbFill.style.backgroundColor = data.is_bot ? '#ef4444' : '#10b981';
            
            cleanedTextP.textContent = `Processed: ${data.cleaned_text}`;

        } catch (error) {
            console.error('Error:', error);
            alert('Prediction failed. Please try again.');
        } finally {
            analyzeBtn.disabled = false;
            loader.style.display = 'none';
            btnText.style.opacity = '1';
        }
    });

    // Add scroll animation observer
    const observerOptions = {
        threshold: 0.1
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('reveal');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    document.querySelectorAll('.reveal').forEach(el => observer.observe(el));

    // Modal Logic
    const modal = document.getElementById("imageModal");
    const modalImg = document.getElementById("fullImage");
    const captionText = document.getElementById("caption");
    const closeBtn = document.querySelector(".modal-close");

    document.querySelectorAll('.dashboard-card img').forEach(img => {
        img.onclick = function() {
            modal.style.display = "block";
            modalImg.src = this.src;
            captionText.innerHTML = this.alt;
        }
    });

    closeBtn.onclick = function() {
        modal.style.display = "none";
    }

    // Close modal when clicking outside the image
    window.onclick = function(event) {
        if (event.target == modal) {
            modal.style.display = "none";
        }
    }
});
