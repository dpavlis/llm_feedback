/**
 * LLM Feedback Chat Application - Frontend Logic
 */

class ChatApp {
    constructor() {
        this.currentConversationId = null;
        this.currentFeedbackMessageId = null;
        this.selectedRating = null;
        this.isLoading = false;
        this.userName = localStorage.getItem('llm_chat_user_name') || '';
        this.manualInputHeight = null;
        this.lastAutoResizeHeight = null;
        this.maxModelLen = window.APP_CONFIG && Number.isFinite(window.APP_CONFIG.maxModelLen)
            ? window.APP_CONFIG.maxModelLen
            : null;

        this.init();
    }

    init() {
        this.bindElements();
        this.bindEvents();
        this.initGenControls();
        this.loadConversations();
        this.autoResizeTextarea();
        this.configureMarked();
    }

    initGenControls() {
        const cfg = window.APP_CONFIG || {};
        const temperature = cfg.defaultTemperature ?? 0.7;
        const topP = cfg.defaultTopP ?? 0.9;
        const topK = cfg.defaultTopK ?? 50;
        const repPenalty = cfg.defaultRepetitionPenalty ?? 1.2;

        this.ctrlTemperature.value = temperature;
        this.ctrlTemperatureVal.textContent = temperature.toFixed(2);
        this.ctrlTopP.value = topP;
        this.ctrlTopPVal.textContent = topP.toFixed(2);
        this.ctrlTopK.value = topK;
        this.ctrlTopKVal.textContent = topK;
        this.ctrlRepPenalty.value = repPenalty;
        this.ctrlRepPenaltyVal.textContent = repPenalty.toFixed(2);
    }

    configureMarked() {
        // Configure marked for safe rendering with syntax highlighting
        marked.setOptions({
            breaks: true,  // Convert \n to <br>
            gfm: true,     // GitHub Flavored Markdown
            highlight: function(code, lang) {
                if (lang && hljs.getLanguage(lang)) {
                    try {
                        return hljs.highlight(code, { language: lang }).value;
                    } catch (e) {
                        console.error('Highlight error:', e);
                    }
                }
                // Auto-detect language if not specified
                try {
                    return hljs.highlightAuto(code).value;
                } catch (e) {
                    return code;
                }
            }
        });
    }

    renderMarkdown(text) {
        // Render markdown to HTML
        try {
            return marked.parse(text);
        } catch (e) {
            console.error('Markdown parsing error:', e);
            return this.escapeHtml(text);
        }
    }

    bindElements() {
        // Main elements
        this.welcomeScreen = document.getElementById('welcome-screen');
        this.chatArea = document.getElementById('chat-area');
        this.messagesArea = document.getElementById('chat-messages');
        this.messageInput = document.getElementById('message-input');
        this.sendBtn = document.getElementById('send-btn');
        this.typingIndicator = document.getElementById('typing-indicator');
        this.conversationList = document.getElementById('conversation-list');

        // Buttons
        this.newConversationBtn = document.getElementById('new-conversation-btn');
        this.welcomeNewBtn = document.getElementById('welcome-new-btn');

        // Feedback Modal elements
        this.feedbackModal = document.getElementById('feedback-modal');
        this.ratingStars = document.querySelectorAll('#rating-stars .star');
        this.feedbackComment = document.getElementById('feedback-comment');
        this.preferredResponse = document.getElementById('preferred-response');
        this.submitFeedbackBtn = document.getElementById('submit-feedback-btn');
        this.cancelFeedbackBtn = document.getElementById('cancel-feedback-btn');
        this.closeModalBtn = document.getElementById('close-modal-btn');

        // New Conversation Modal elements
        this.newConvModal = document.getElementById('new-conversation-modal');
        this.userNameInput = document.getElementById('user-name-input');
        this.startConversationBtn = document.getElementById('start-conversation-btn');
        this.cancelNewConvBtn = document.getElementById('cancel-new-conv-btn');
        this.closeNewConvBtn = document.getElementById('close-new-conv-btn');

        // Toast
        this.errorToast = document.getElementById('error-toast');
        this.errorMessage = document.getElementById('error-message');

        // Generation controls
        this.ctrlTemperature = document.getElementById('ctrl-temperature');
        this.ctrlTemperatureVal = document.getElementById('ctrl-temperature-val');
        this.ctrlTopP = document.getElementById('ctrl-top-p');
        this.ctrlTopPVal = document.getElementById('ctrl-top-p-val');
        this.ctrlTopK = document.getElementById('ctrl-top-k');
        this.ctrlTopKVal = document.getElementById('ctrl-top-k-val');
        this.ctrlRepPenalty = document.getElementById('ctrl-rep-penalty');
        this.ctrlRepPenaltyVal = document.getElementById('ctrl-rep-penalty-val');
    }

    bindEvents() {
        // New conversation buttons - now open modal instead of creating directly
        this.newConversationBtn.addEventListener('click', () => this.openNewConversationModal());
        this.welcomeNewBtn.addEventListener('click', () => this.openNewConversationModal());

        // Send message
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Enable/disable send button based on input
        this.messageInput.addEventListener('input', () => {
            this.sendBtn.disabled = this.messageInput.value.trim() === '' || this.isLoading;
            this.autoResizeTextarea();
        });

        this.messageInput.addEventListener('mouseup', () => this.captureManualResize());
        this.messageInput.addEventListener('touchend', () => this.captureManualResize());

        // Rating stars
        this.ratingStars.forEach(star => {
            star.addEventListener('click', () => this.setRating(parseInt(star.dataset.rating)));
            star.addEventListener('mouseenter', () => this.highlightStars(parseInt(star.dataset.rating)));
        });

        document.getElementById('rating-stars').addEventListener('mouseleave', () => {
            this.highlightStars(this.selectedRating || 0);
        });

        // Feedback Modal buttons
        this.submitFeedbackBtn.addEventListener('click', () => this.submitFeedback());
        this.cancelFeedbackBtn.addEventListener('click', () => this.closeFeedbackModal());
        this.closeModalBtn.addEventListener('click', () => this.closeFeedbackModal());
        this.feedbackModal.querySelector('.modal-overlay').addEventListener('click', () => this.closeFeedbackModal());

        // New Conversation Modal buttons
        this.startConversationBtn.addEventListener('click', () => this.createConversationFromModal());
        this.cancelNewConvBtn.addEventListener('click', () => this.closeNewConversationModal());
        this.closeNewConvBtn.addEventListener('click', () => this.closeNewConversationModal());
        this.newConvModal.querySelector('.modal-overlay').addEventListener('click', () => this.closeNewConversationModal());

        // Allow Enter to start conversation in name input
        this.userNameInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                this.createConversationFromModal();
            }
        });

        // Generation control sliders
        this.ctrlTemperature.addEventListener('input', () => {
            this.ctrlTemperatureVal.textContent = parseFloat(this.ctrlTemperature.value).toFixed(2);
        });
        this.ctrlTopP.addEventListener('input', () => {
            this.ctrlTopPVal.textContent = parseFloat(this.ctrlTopP.value).toFixed(2);
        });
        this.ctrlTopK.addEventListener('input', () => {
            this.ctrlTopKVal.textContent = this.ctrlTopK.value;
        });
        this.ctrlRepPenalty.addEventListener('input', () => {
            this.ctrlRepPenaltyVal.textContent = parseFloat(this.ctrlRepPenalty.value).toFixed(2);
        });
    }

    autoResizeTextarea() {
        this.messageInput.style.height = 'auto';
        const autoHeight = Math.min(this.messageInput.scrollHeight, 150);
        const targetHeight = this.manualInputHeight
            ? Math.max(autoHeight, this.manualInputHeight)
            : autoHeight;
        this.messageInput.style.height = targetHeight + 'px';
        this.lastAutoResizeHeight = targetHeight;
    }

    captureManualResize() {
        const currentHeight = this.messageInput.offsetHeight;
        if (this.lastAutoResizeHeight === null) {
            return;
        }
        if (Math.abs(currentHeight - this.lastAutoResizeHeight) > 1) {
            this.manualInputHeight = currentHeight;
        }
    }

    async loadConversations() {
        try {
            const response = await fetch('/api/conversations');
            const data = await response.json();

            this.renderConversationList(data.conversations);
        } catch (error) {
            console.error('Failed to load conversations:', error);
        }
    }

    renderConversationList(conversations) {
        this.conversationList.innerHTML = '';

        if (conversations.length === 0) {
            return;
        }

        conversations.forEach(conv => {
            const item = document.createElement('div');
            item.className = 'conversation-item';
            if (conv.conversation_id === this.currentConversationId) {
                item.classList.add('active');
            }

            const preview = conv.preview || 'New conversation';
            const date = new Date(conv.created_at);
            const dateStr = date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

            const tokenCount = Number.isFinite(conv.token_count)
                ? conv.token_count
                : 0;
            const tokenText = tokenCount.toLocaleString();
            const breakdownText = this.formatTokenBreakdown(conv.token_breakdown);
            const status = this.getTokenStatus(tokenCount);
            const statusMarkup = status
                ? `<span class="token-status ${status.className}" title="${status.title}">${status.label}</span> `
                : '';
            const metaText = `${conv.message_count} messages`;
            const breakdownMarkup = breakdownText ? ` · ${breakdownText}` : '';

            item.innerHTML = `
                <div class="preview">${this.escapeHtml(preview)}</div>
                <div class="meta">
                    <div class="meta-line">${metaText}</div>
                    <div class="meta-line">${statusMarkup}${tokenText} tokens${breakdownMarkup}</div>
                    <div class="meta-line">${dateStr}</div>
                </div>
                <button class="copy-conv-btn" title="Copy conversation as JSON">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
                </button>
                <button class="delete-btn" title="Delete conversation">&times;</button>
            `;

            // Click on item to select conversation
            item.addEventListener('click', (e) => {
                if (!e.target.classList.contains('delete-btn') && !e.target.closest('.copy-conv-btn')) {
                    this.selectConversation(conv.conversation_id);
                }
            });

            // Click on copy conversation button
            const copyConvBtn = item.querySelector('.copy-conv-btn');
            copyConvBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.copyConversationAsJson(conv.conversation_id, copyConvBtn);
            });

            // Click on delete button
            const deleteBtn = item.querySelector('.delete-btn');
            deleteBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.deleteConversation(conv.conversation_id);
            });

            this.conversationList.appendChild(item);
        });
    }

    formatTokenBreakdown(breakdown) {
        if (!breakdown || typeof breakdown !== 'object') {
            return '';
        }
        const system = Number.isFinite(breakdown.system) ? breakdown.system : 0;
        const user = Number.isFinite(breakdown.user) ? breakdown.user : 0;
        const assistant = Number.isFinite(breakdown.assistant) ? breakdown.assistant : 0;
        if (system === 0 && user === 0 && assistant === 0) {
            return '';
        }
        return `sys ${system.toLocaleString()} · user ${user.toLocaleString()} · asst ${assistant.toLocaleString()}`;
    }

    getTokenStatus(tokenCount) {
        if (!this.maxModelLen || !Number.isFinite(tokenCount)) {
            return null;
        }
        if (tokenCount >= this.maxModelLen) {
            return { label: 'X', className: 'token-error', title: 'Context limit exceeded' };
        }
        if (tokenCount >= this.maxModelLen * 0.9) {
            return { label: '!', className: 'token-warning', title: 'Approaching context limit' };
        }
        return null;
    }

    async selectConversation(conversationId) {
        try {
            const response = await fetch(`/api/conversations/${conversationId}`);
            if (!response.ok) {
                throw new Error('Failed to load conversation');
            }

            const data = await response.json();
            this.currentConversationId = conversationId;

            // Show chat area
            this.welcomeScreen.classList.add('hidden');
            this.chatArea.classList.remove('hidden');

            // Clear and render messages
            this.messagesArea.innerHTML = '';
            data.messages.forEach(msg => {
                this.appendMessage(msg.role, msg.content, msg.id, msg.feedback, msg.generation_ms);
            });

            // Update conversation list active state
            this.loadConversations();

            // Scroll to bottom
            this.scrollToBottom();
        } catch (error) {
            this.showError('Failed to load conversation');
            console.error(error);
        }
    }

    // New Conversation Modal
    openNewConversationModal() {
        // Pre-fill with saved name
        this.userNameInput.value = this.userName;
        this.newConvModal.classList.remove('hidden');
        this.userNameInput.focus();
    }

    closeNewConversationModal() {
        this.newConvModal.classList.add('hidden');
    }

    async createConversationFromModal() {
        // Save the user name to localStorage
        const name = this.userNameInput.value.trim();
        this.userName = name;
        localStorage.setItem('llm_chat_user_name', name);

        this.closeNewConversationModal();
        await this.newConversation(name);
    }

    async newConversation(userName = null) {
        try {
            const response = await fetch('/api/conversations', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    user_name: userName || this.userName || null
                }),
            });

            if (!response.ok) {
                throw new Error('Failed to create conversation');
            }

            const data = await response.json();
            this.currentConversationId = data.conversation_id;

            // Show chat area
            this.welcomeScreen.classList.add('hidden');
            this.chatArea.classList.remove('hidden');

            // Clear messages
            this.messagesArea.innerHTML = '';

            // Reload conversation list
            this.loadConversations();

            // Focus input
            this.messageInput.focus();
        } catch (error) {
            this.showError('Failed to create new conversation');
            console.error(error);
        }
    }

    async deleteConversation(conversationId) {
        if (!confirm('Delete this conversation from the list? (The conversation history will still be saved on the server)')) {
            return;
        }

        try {
            const response = await fetch(`/api/conversations/${conversationId}`, {
                method: 'DELETE',
            });

            if (!response.ok) {
                throw new Error('Failed to delete conversation');
            }

            // If we deleted the current conversation, show welcome screen
            if (conversationId === this.currentConversationId) {
                this.currentConversationId = null;
                this.welcomeScreen.classList.remove('hidden');
                this.chatArea.classList.add('hidden');
            }

            // Reload conversation list
            this.loadConversations();
        } catch (error) {
            this.showError('Failed to delete conversation');
            console.error(error);
        }
    }

    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || this.isLoading) return;

        // Ensure we have a conversation
        if (!this.currentConversationId) {
            await this.newConversation();
        }

        // Clear input and disable
        this.messageInput.value = '';
        this.messageInput.style.height = 'auto';
        this.sendBtn.disabled = true;
        this.isLoading = true;

        // Display user message
        this.appendMessage('user', message);

        // Show typing indicator
        this.showTypingIndicator();

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    conversation_id: this.currentConversationId,
                    message: message,
                    temperature: parseFloat(this.ctrlTemperature.value),
                    top_p: parseFloat(this.ctrlTopP.value),
                    top_k: parseInt(this.ctrlTopK.value, 10),
                    repetition_penalty: parseFloat(this.ctrlRepPenalty.value),
                }),
            });

            if (!response.ok) {
                throw new Error('Failed to send message');
            }

            const data = await response.json();
            this.hideTypingIndicator();
            this.appendMessage('assistant', data.response, data.message_id, null, data.generation_ms);

            // Reload conversation list to update preview
            this.loadConversations();
        } catch (error) {
            this.hideTypingIndicator();
            this.showError('Failed to send message. Please try again.');
            console.error(error);
        } finally {
            this.isLoading = false;
            this.sendBtn.disabled = this.messageInput.value.trim() === '';
        }
    }

    appendMessage(role, content, messageId = null, feedback = null, generationMs = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';

        // Render markdown for assistant messages, plain text for user messages
        if (role === 'assistant') {
            contentDiv.innerHTML = this.renderMarkdown(content);
        } else {
            contentDiv.textContent = content;
        }
        messageDiv.appendChild(contentDiv);

        // Add actions row (copy button + feedback button)
        const actionsDiv = document.createElement('div');
        actionsDiv.className = 'message-actions';

        // Copy button for all messages
        const copyBtn = document.createElement('button');
        copyBtn.className = 'copy-msg-btn';
        copyBtn.title = 'Copy to clipboard';
        copyBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
        copyBtn.addEventListener('click', () => this.copyMessageToClipboard(content, copyBtn));
        actionsDiv.appendChild(copyBtn);

        // Add feedback button for assistant messages
        if (role === 'assistant' && messageId) {
            const feedbackBtn = document.createElement('button');
            feedbackBtn.className = 'feedback-btn';
            if (feedback) {
                feedbackBtn.classList.add('submitted');
                feedbackBtn.textContent = `✓ Rated ${feedback.rating || '-'}/5`;
            } else {
                feedbackBtn.textContent = '👍 Rate this response';
            }
            feedbackBtn.addEventListener('click', () => this.openFeedbackModal(messageId, feedback));
            actionsDiv.appendChild(feedbackBtn);

            if (Number.isFinite(generationMs)) {
                const genTime = document.createElement('span');
                genTime.className = 'gen-time';
                genTime.textContent = `⏱ ${this.formatDurationMs(generationMs)}`;
                actionsDiv.appendChild(genTime);
            }
        }

        messageDiv.appendChild(actionsDiv);

        this.messagesArea.appendChild(messageDiv);
        this.scrollToBottom();
    }

    formatDurationMs(durationMs) {
        const totalSeconds = Math.max(0, Math.round(durationMs / 1000));
        const minutes = Math.floor(totalSeconds / 60);
        const seconds = totalSeconds % 60;
        return `${minutes}:${seconds.toString().padStart(2, '0')}`;
    }

    async copyMessageToClipboard(content, btn) {
        try {
            const copied = await this.writeToClipboard(content);
            if (!copied) {
                throw new Error('Clipboard write failed');
            }
            btn.classList.add('copied');
            btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';
            setTimeout(() => {
                btn.classList.remove('copied');
                btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
            }, 2000);
        } catch (e) {
            console.error('Failed to copy:', e);
            this.showError('Failed to copy message');
        }
    }

    async copyConversationAsJson(conversationId, btn) {
        try {
            const response = await fetch(`/api/conversations/${conversationId}`);
            if (!response.ok) throw new Error('Failed to load conversation');
            const data = await response.json();

            const messages = data.messages.map(msg => {
                if (msg.role === 'user') {
                    return { user: msg.content };
                } else {
                    const entry = { assistant: msg.content };
                    if (msg.feedback) {
                        if (msg.feedback.comment) entry.user_comment = msg.feedback.comment;
                        if (msg.feedback.preferred_response) entry.suggestion = msg.feedback.preferred_response;
                    }
                    return entry;
                }
            });

            const json = JSON.stringify({ messages }, null, 2);
            const copied = await this.writeToClipboard(json);
            if (!copied) {
                throw new Error('Clipboard write failed');
            }

            btn.classList.add('copied');
            btn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';
            setTimeout(() => {
                btn.classList.remove('copied');
                btn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
            }, 2000);
        } catch (e) {
            console.error('Failed to copy conversation:', e);
            this.showError('Failed to copy conversation');
        }
    }

    async writeToClipboard(text) {
        if (navigator.clipboard && window.isSecureContext) {
            await navigator.clipboard.writeText(text);
            return true;
        }

        const textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.setAttribute('readonly', '');
        textarea.style.position = 'fixed';
        textarea.style.top = '-9999px';
        textarea.style.left = '-9999px';
        document.body.appendChild(textarea);

        textarea.select();
        textarea.setSelectionRange(0, textarea.value.length);
        let copied = false;

        try {
            copied = document.execCommand('copy');
        } catch (e) {
            copied = false;
        } finally {
            document.body.removeChild(textarea);
        }

        return copied;
    }

    showTypingIndicator() {
        this.typingIndicator.classList.remove('hidden');
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        this.typingIndicator.classList.add('hidden');
    }

    scrollToBottom() {
        this.messagesArea.scrollTop = this.messagesArea.scrollHeight;
    }

    // Feedback Modal
    openFeedbackModal(messageId, feedback = null) {
        this.currentFeedbackMessageId = messageId;
        this.selectedRating = feedback?.rating || null;
        this.highlightStars(this.selectedRating || 0);
        this.feedbackComment.value = feedback?.comment || '';
        this.preferredResponse.value = feedback?.preferred_response || '';
        this.feedbackModal.classList.remove('hidden');
    }

    closeFeedbackModal() {
        this.feedbackModal.classList.add('hidden');
        this.currentFeedbackMessageId = null;
        this.selectedRating = null;
    }

    setRating(rating) {
        this.selectedRating = rating;
        this.highlightStars(rating);
    }

    highlightStars(rating) {
        this.ratingStars.forEach(star => {
            const starRating = parseInt(star.dataset.rating);
            if (starRating <= rating) {
                star.classList.add('active');
            } else {
                star.classList.remove('active');
            }
        });
    }

    async submitFeedback() {
        if (!this.currentFeedbackMessageId) return;

        const feedback = {
            message_id: this.currentFeedbackMessageId,
            rating: this.selectedRating,
            comment: this.feedbackComment.value.trim() || null,
            preferred_response: this.preferredResponse.value.trim() || null,
        };

        try {
            const response = await fetch(`/api/conversations/${this.currentConversationId}/feedback`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(feedback),
            });

            if (!response.ok) {
                throw new Error('Failed to submit feedback');
            }

            this.closeFeedbackModal();

            // Update the feedback button to show submitted state
            const feedbackBtns = this.messagesArea.querySelectorAll('.feedback-btn');
            feedbackBtns.forEach(btn => {
                // Find the button for this message (we'll need to track this better in a real app)
                if (!btn.classList.contains('submitted')) {
                    btn.classList.add('submitted');
                    btn.textContent = `✓ Rated ${this.selectedRating || '-'}/5`;
                }
            });

            // Reload conversation to get updated feedback
            this.selectConversation(this.currentConversationId);
        } catch (error) {
            this.showError('Failed to submit feedback');
            console.error(error);
        }
    }

    // Error handling
    showError(message) {
        this.errorMessage.textContent = message;
        this.errorToast.classList.remove('hidden');

        setTimeout(() => {
            this.errorToast.classList.add('hidden');
        }, 5000);
    }

    // Utility
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize the app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.chatApp = new ChatApp();
});
