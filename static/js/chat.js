/**
 * ResearchMate AI - Interactive RAG Chat Client
 * Handles real-time question dispatch, loading states, and clickable citations.
 */

document.addEventListener('DOMContentLoaded', function () {
  const chatForm = document.getElementById('chatForm');
  const chatInput = document.getElementById('chatInput');
  const chatMessages = document.getElementById('chatMessages');
  const conversationId = chatForm ? chatForm.dataset.conversationId : null;

  if (!chatForm || !conversationId) return;

  function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }
  scrollToBottom();

  // Auto-resize textarea
  chatInput.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
  });

  // Submit on Enter without Shift
  chatInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      chatForm.dispatchEvent(new Event('submit'));
    }
  });

  chatForm.addEventListener('submit', async function (e) {
    e.preventDefault();
    const question = chatInput.value.trim();
    if (!question) return;

    // Append User Message to UI immediately
    appendMessage('user', question);
    chatInput.value = '';
    chatInput.style.height = 'auto';
    chatInput.disabled = true;

    // Append Loading Indicator
    const loadingElem = appendLoadingMessage();
    scrollToBottom();

    // Get CSRF Token
    const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]').value;

    try {
      const response = await fetch(`/chat/${conversationId}/send/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': csrfToken,
        },
        body: JSON.stringify({ question: question }),
      });

      const data = await response.json();
      loadingElem.remove();

      if (response.ok && data.status === 'success') {
        appendMessage('assistant', data.answer, data.citations, data.has_sufficient_evidence);
      } else {
        appendErrorMessage(data.error || 'Failed to generate answer from indexed documents.');
      }
    } catch (err) {
      loadingElem.remove();
      appendErrorMessage('Network or server error while connecting to research assistant.');
    } finally {
      chatInput.disabled = false;
      chatInput.focus();
      scrollToBottom();
    }
  });

  function appendMessage(role, content, citations = [], hasSufficientEvidence = true) {
    const bubble = document.createElement('div');
    bubble.className = `message-bubble message-${role}`;

    const roleLabel = document.createElement('div');
    roleLabel.className = 'message-role-label';
    roleLabel.textContent = role === 'user' ? 'You' : 'ResearchMate AI';
    bubble.appendChild(roleLabel);

    const textElem = document.createElement('div');
    textElem.style.whiteSpace = 'pre-wrap';
    textElem.textContent = content;
    bubble.appendChild(textElem);

    // If assistant and citations exist, render citations tray
    if (role === 'assistant' && citations && citations.length > 0) {
      const citationsContainer = document.createElement('div');
      citationsContainer.className = 'citations-container';

      const header = document.createElement('div');
      header.className = 'citations-header';
      header.innerHTML = `<span>📑 Grounded Citations (${citations.length})</span>`;
      citationsContainer.appendChild(header);

      const chips = document.createElement('div');
      chips.className = 'citation-chips';

      citations.forEach((cit, idx) => {
        const chip = document.createElement('a');
        chip.className = 'citation-chip';
        // Link directly to authorized PDF viewer with page hash
        const pageNum = (cit.page_reference || '1').toString().split('-')[0];
        chip.href = `/documents/${encodeURIComponent(cit.document_id)}/view/#page=${encodeURIComponent(pageNum)}`;
        chip.target = '_blank';
        chip.rel = 'noopener noreferrer';
        chip.title = `Excerpt: ${cit.excerpt || ''}`;
        
        const labelSpan = document.createElement('span');
        labelSpan.textContent = `📄 [Source ${idx + 1}] ${cit.filename} (p. ${cit.page_reference})`;
        chip.appendChild(labelSpan);
        chips.appendChild(chip);
      });

      citationsContainer.appendChild(chips);
      bubble.appendChild(citationsContainer);
    } else if (role === 'assistant' && !hasSufficientEvidence) {
      const notice = document.createElement('div');
      notice.style.marginTop = '0.75rem';
      notice.style.fontSize = '0.825rem';
      notice.style.color = '#92400e';
      notice.style.background = '#fffbeb';
      notice.style.padding = '0.4rem 0.6rem';
      notice.style.borderRadius = '4px';
      notice.innerHTML = `ℹ️ <em>Retrieved similarity fell below threshold. Evidence limitation communicated.</em>`;
      bubble.appendChild(notice);
    }

    chatMessages.appendChild(bubble);
    scrollToBottom();
  }

  function appendLoadingMessage() {
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble message-assistant';
    bubble.id = 'loadingBubble';

    const roleLabel = document.createElement('div');
    roleLabel.className = 'message-role-label';
    roleLabel.textContent = 'ResearchMate AI';
    bubble.appendChild(roleLabel);

    const indicator = document.createElement('div');
    indicator.style.display = 'flex';
    indicator.style.alignItems = 'center';
    indicator.style.gap = '0.5rem';
    indicator.innerHTML = `
      <span style="display:inline-block; animation: pulse 1.5s infinite;">🔍</span>
      <span>Retrieving authorized document evidence and generating grounded response...</span>
    `;
    bubble.appendChild(indicator);

    chatMessages.appendChild(bubble);
    return bubble;
  }

  function appendErrorMessage(errorMsg) {
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble message-assistant';
    bubble.style.borderLeft = '4px solid #ef4444';

    const roleLabel = document.createElement('div');
    roleLabel.className = 'message-role-label';
    roleLabel.style.color = '#ef4444';
    roleLabel.textContent = 'Assistant Error';
    bubble.appendChild(roleLabel);

    const text = document.createElement('div');
    text.textContent = errorMsg;
    bubble.appendChild(text);

    chatMessages.appendChild(bubble);
  }
});
