// Study Buddy - Web Dashboard Frontend Application Logic
let activeSubjectSlug = "";
let activeSubjectName = "";
let activeTopicName = "";
let sessionId = 1; // Default session
let isAsking = false;

// 1. Initializer on DOM Load
document.addEventListener("DOMContentLoaded", () => {
    initApp();
});

async function initApp() {
    setupTabSwitching();
    setupEventListeners();
    await bootstrapSession();
    await loadSubjects();
    updateBackendStatus(true);
}

// 2. Tab Navigation
function setupTabSwitching() {
    const navButtons = document.querySelectorAll(".nav-btn");
    const tabPanes = document.querySelectorAll(".tab-pane");
    const currentTabTitle = document.getElementById("current-tab-title");

    navButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            navButtons.forEach(b => b.classList.remove("active"));
            tabPanes.forEach(pane => pane.classList.remove("active"));

            btn.classList.add("active");
            const targetTab = btn.getAttribute("data-tab");
            document.getElementById(targetTab).classList.add("active");

            // Update Header Title
            currentTabTitle.textContent = btn.textContent.trim().replace(/^[\u2000-\u2BFF\uE000-\uF8FF]/, '').trim();
        });
    });
}

// 3. API Communication Helpers
async function apiCall(endpoint, method = "GET", body = null) {
    const options = {
        method,
        headers: {
            "Content-Type": "application/json"
        }
    };
    if (body) {
        options.body = JSON.stringify(body);
    }
    
    try {
        const response = await fetch(endpoint, options);
        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || `HTTP error! status: ${response.status}`);
        }
        return await response.json();
    } catch (e) {
        console.error("API Call Failed:", e);
        throw e;
    }
}

function updateBackendStatus(connected, text = "Connected") {
    const dot = document.querySelector(".status-dot");
    const label = document.getElementById("backend-status");
    if (connected) {
        dot.style.backgroundColor = "var(--color-success)";
        dot.style.boxShadow = "0 0 8px var(--color-success)";
        label.textContent = text;
    } else {
        dot.style.backgroundColor = "var(--color-error)";
        dot.style.boxShadow = "0 0 8px var(--color-error)";
        label.textContent = text;
    }
}

// 4. Session Bootstream
async function bootstrapSession() {
    try {
        const sessions = await apiCall("/sessions");
        if (sessions && sessions.length > 0) {
            sessionId = sessions[0].id;
        } else {
            const result = await apiCall("/session", "POST", { title: "My Study Buddy Session" });
            sessionId = result.session_id;
        }
    } catch (e) {
        showGlobalError("Cannot connect to backend uvicorn server. Make sure it is running on port 8000!");
        updateBackendStatus(false, "Offline");
    }
}

// 5. Subjects Explainer Logic
async function loadSubjects() {
    try {
        const subjects = await apiCall("/subjects");
        const selectorList = document.getElementById("subject-selector-list");
        const tbody = document.getElementById("ingest-status-tbody");
        
        selectorList.innerHTML = "";
        tbody.innerHTML = "";

        if (subjects.length === 0) {
            selectorList.innerHTML = '<p class="placeholder-text">No subjects found.</p>';
            tbody.innerHTML = '<tr><td colspan="4" class="placeholder-text">No subjects configured in system.</td></tr>';
            return;
        }

        subjects.forEach(subj => {
            // Render subject button in explainer panel
            const btn = document.createElement("button");
            btn.className = "subject-btn";
            btn.textContent = subj.subject;
            btn.setAttribute("data-slug", subj.slug);
            btn.addEventListener("click", () => {
                document.querySelectorAll(".subject-btn").forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                selectSubject(subj.slug, subj.subject);
            });
            selectorList.appendChild(btn);

            // Render row in Ingestion Settings
            const tr = document.createElement("tr");
            const isIndexed = subj.total_chunks > 0;
            const statusText = isIndexed ? `<span class="badge" style="background-color: rgba(16, 185, 129, 0.15); color: var(--color-success)">Indexed</span>` : `<span class="badge" style="background-color: rgba(239, 68, 68, 0.15); color: var(--color-error)">Unindexed</span>`;
            
            tr.innerHTML = `
                <td><strong>${subj.subject}</strong></td>
                <td id="ingest-status-${subj.slug}">${statusText}</td>
                <td id="ingest-count-${subj.slug}">${subj.topics_total} topics (${subj.total_chunks} chunks)</td>
                <td>
                    <button class="btn btn-sm btn-ingest" id="btn-ingest-${subj.slug}" onclick="triggerIngestion('${subj.slug}')">
                        ${isIndexed ? 'Re-Index' : 'Index'}
                    </button>
                    <div class="progress-bar-container" id="progress-container-${subj.slug}" style="display:none; margin-top: 6px; height: 4px;">
                        <div class="progress-bar-fill" id="progress-bar-${subj.slug}" style="width: 0%"></div>
                    </div>
                </td>
            `;
            tbody.appendChild(tr);
        });

        // Pre-select first subject if none active
        if (!activeSubjectSlug && subjects.length > 0) {
            selectorList.children[0].click();
        }
    } catch (e) {
        showGlobalError("Failed to fetch subjects list.");
    }
}

async function selectSubject(slug, name) {
    activeSubjectSlug = slug;
    activeSubjectName = name;
    activeTopicName = "";
    
    document.getElementById("selected-subject-name").textContent = name;
    document.getElementById("topics-progress-label").textContent = "Loading...";
    document.getElementById("topics-progress-bar").style.width = "0%";
    
    // Hide chat input and export button until a topic is chosen
    document.getElementById("subjects-chat-input-container").style.display = "none";
    document.getElementById("btn-export-subjects").style.display = "none";
    
    const checklistDiv = document.getElementById("topics-checklist");
    checklistDiv.innerHTML = '<p class="placeholder-text">Loading checklist...</p>';

    const viewport = document.getElementById("explanation-body");
    viewport.innerHTML = `
        <div class="welcome-placeholder">
            <span class="placeholder-icon">📖</span>
            <p>Select any topic from the checklist on the left, and Qwen 3 will pull concepts from the indexed PDF database and explain them in elaborative detail here.</p>
        </div>
    `;

    try {
        const data = await apiCall(`/subjects/${slug}/topics`);
        renderTopics(data.topics, data.progress);
    } catch (e) {
        checklistDiv.innerHTML = '<p class="placeholder-text" style="color:var(--color-error)">Error loading subject topics checklist.</p>';
    }
}

function renderTopics(topics, progress) {
    const checklistDiv = document.getElementById("topics-checklist");
    const progressLabel = document.getElementById("topics-progress-label");
    const progressBar = document.getElementById("topics-progress-bar");
    
    checklistDiv.innerHTML = "";

    if (!topics || topics.length === 0) {
        progressLabel.textContent = "0 / 0 done";
        progressBar.style.width = "0%";
        checklistDiv.innerHTML = `<p class="placeholder-text">No topics extracted yet.<br><br>Please trigger indexing for <strong>${activeSubjectName}</strong> in the Index & Settings tab above.</p>`;
        return;
    }

    // Set progress bar
    const doneCount = progress.done;
    const totalCount = progress.total;
    const pct = totalCount > 0 ? (doneCount / totalCount) * 100 : 0;
    
    progressLabel.textContent = `${doneCount} / ${totalCount} done`;
    progressBar.style.width = `${pct}%`;

    topics.forEach((t, i) => {
        const row = document.createElement("div");
        row.className = `topic-row ${t.done ? 'done' : ''}`;
        if (activeTopicName === t.name) {
            row.classList.add("active-topic");
        }
        
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.className = "topic-checkbox";
        checkbox.checked = t.done;
        checkbox.addEventListener("change", () => {
            toggleTopic(t.name, checkbox.checked);
        });

        const btn = document.createElement("button");
        btn.className = "topic-select-btn";
        btn.textContent = `${i + 1}. ${t.name}`;
        btn.title = `Click to explain ${t.name}`;
        btn.addEventListener("click", () => {
            document.querySelectorAll(".topic-row").forEach(r => r.classList.remove("active-topic"));
            row.classList.add("active-topic");
            explainTopic(t.name);
        });

        row.appendChild(checkbox);
        row.appendChild(btn);
        checklistDiv.appendChild(row);
    });
}

async function toggleTopic(topicName, done) {
    try {
        const data = await apiCall(`/subjects/${activeSubjectSlug}/topics/toggle`, "POST", {
            topic: topicName,
            done
        });
        renderTopics(data.topics, data.progress);
    } catch (e) {
        showGlobalError("Failed to toggle topic completion status.");
    }
}

async function explainTopic(topicName) {
    if (isAsking) return;
    activeTopicName = topicName;

    const viewport = document.getElementById("explanation-body");
    const modeStr = `subject:${activeSubjectSlug}:${topicName}`;

    viewport.innerHTML = `<div class="welcome-placeholder"><p>Loading topic discussion...</p></div>`;
    
    try {
        const history = await apiCall(`/session/${sessionId}/messages?mode=${modeStr}`);
        
        if (history && history.length > 0) {
            renderExplanationHistory(history);
            return;
        }
        
        setAskingState(true, "explanation");
        viewport.innerHTML = `
            <div class="welcome-placeholder">
                <span class="placeholder-icon">🤖</span>
                <p style="color: var(--accent-hover); font-weight: 500;">Retrieving subject ebook context for "${topicName}" and generating detailed lesson explanation...</p>
            </div>
        `;

        await apiCall(`/subjects/${activeSubjectSlug}/explain_topic`, "POST", {
            session_id: sessionId,
            topic: topicName
        });
        
        const data = await apiCall(`/subjects/${activeSubjectSlug}/topics`);
        renderTopics(data.topics, data.progress);
        
        const newHistory = await apiCall(`/session/${sessionId}/messages?mode=${modeStr}`);
        renderExplanationHistory(newHistory);
    } catch (e) {
        viewport.innerHTML = `
            <div class="welcome-placeholder" style="color:var(--color-error)">
                <span class="placeholder-icon">❌</span>
                <p>Failed to explain topic. Make sure uvicorn and your local Ollama model are running correctly.</p>
            </div>
        `;
    } finally {
        setAskingState(false, "explanation");
    }
}

function renderExplanationHistory(history) {
    const viewport = document.getElementById("explanation-body");
    const chatInputContainer = document.getElementById("subjects-chat-input-container");
    const exportBtn = document.getElementById("btn-export-subjects");
    
    viewport.innerHTML = "";

    if (!history || history.length === 0) {
        chatInputContainer.style.display = "none";
        exportBtn.style.display = "none";
        viewport.innerHTML = `
            <div class="welcome-placeholder">
                <span class="placeholder-icon">📖</span>
                <p>Select any topic from the checklist on the left, and Qwen 3 will pull concepts from the indexed PDF database and explain them in elaborative detail here.</p>
            </div>
        `;
        return;
    }

    chatInputContainer.style.display = "flex";
    exportBtn.style.display = "inline-flex";
    exportBtn.href = `/session/${sessionId}/export?mode=subject:${activeSubjectSlug}:${activeTopicName}`;

    history.forEach(msg => {
        const titleDiv = document.createElement("h2");
        titleDiv.className = "history-question-title";
        
        if (msg.question.startsWith("Explain topic:")) {
            titleDiv.textContent = `📌 Initial Explanation: ${activeTopicName}`;
        } else {
            titleDiv.textContent = `🙋 Follow-up: ${msg.question}`;
        }
        viewport.appendChild(titleDiv);

        const cardDiv = document.createElement("div");
        cardDiv.className = "history-card-body";
        
        const mdText = msg.card.text || msg.card.concept || msg.card.what_is_it || "";
        cardDiv.innerHTML = parseMarkdown(mdText);
        
        if (msg.card.sources && msg.card.sources.length > 0) {
            const sourcesDiv = document.createElement("div");
            sourcesDiv.className = "sources-box";
            sourcesDiv.innerHTML = `<span class="sources-title">📚 Cites & Library Sources:</span>`;
            const ul = document.createElement("ul");
            msg.card.sources.forEach(src => {
                ul.innerHTML += `<li><code>${src.source}</code> — ${src.note}</li>`;
            });
            sourcesDiv.appendChild(ul);
            cardDiv.appendChild(sourcesDiv);
        }

        viewport.appendChild(cardDiv);
    });

    setTimeout(() => {
        viewport.scrollTop = viewport.scrollHeight;
    }, 50);
}

async function sendSubjectFollowup() {
    const input = document.getElementById("subjects-followup-input");
    const question = input.value.trim();
    if (!question) return;
    if (isAsking) return;
    
    setAskingState(true, "explanation");
    input.value = "";
    
    const viewport = document.getElementById("explanation-body");
    const titleDiv = document.createElement("h2");
    titleDiv.className = "history-question-title";
    titleDiv.textContent = `🙋 Follow-up: ${question}`;
    viewport.appendChild(titleDiv);
    
    const loadingDiv = document.createElement("div");
    loadingDiv.className = "history-card-body";
    loadingDiv.innerHTML = `<p style="color: var(--accent-hover); font-style: italic;">Thinking...</p>`;
    viewport.appendChild(loadingDiv);
    viewport.scrollTop = viewport.scrollHeight;

    try {
        await apiCall(`/subjects/${activeSubjectSlug}/followup`, "POST", {
            session_id: sessionId,
            topic: activeTopicName,
            question: question
        });
        
        const modeStr = `subject:${activeSubjectSlug}:${activeTopicName}`;
        const history = await apiCall(`/session/${sessionId}/messages?mode=${modeStr}`);
        renderExplanationHistory(history);
    } catch (e) {
        loadingDiv.innerHTML = `<p style="color: var(--color-error)">Failed to get answer. Please check backend server status.</p>`;
    } finally {
        setAskingState(false, "explanation");
    }
}

// 6. LeetCode Solver Logic
async function solveLeetCode() {
    const problemText = document.getElementById("leetcode-problem-input").value.trim();
    if (!problemText) {
        showGlobalError("Please paste a LeetCode problem statement first.");
        return;
    }
    if (isAsking) return;
    setAskingState(true, "leetcode");

    const viewport = document.getElementById("leetcode-explanation-body");
    viewport.innerHTML = `
        <div class="welcome-placeholder">
            <span class="placeholder-icon">🤖</span>
            <p style="color: var(--accent-hover); font-weight: 500;">Qwen is generating brute-force and optimized code solutions. Study Buddy is compiling deep pattern explanations... Please wait (this can take 30-45 seconds).</p>
        </div>
    `;

    try {
        await apiCall("/leetcode", "POST", {
            session_id: sessionId,
            problem: problemText
        });
        
        const history = await apiCall(`/session/${sessionId}/messages?mode=leetcode`);
        renderLeetCodeHistory(history);
    } catch (e) {
        viewport.innerHTML = `
            <div class="welcome-placeholder" style="color:var(--color-error)">
                <span class="placeholder-icon">❌</span>
                <p>Failed to generate LeetCode solutions. Ensure Ollama models are running correctly.</p>
            </div>
        `;
    } finally {
        setAskingState(false, "leetcode");
    }
}

function renderLeetCodeHistory(history) {
    const viewport = document.getElementById("leetcode-explanation-body");
    const exportBtn = document.getElementById("btn-export-leetcode");
    viewport.innerHTML = "";

    if (!history || history.length === 0) {
        exportBtn.style.display = "none";
        viewport.innerHTML = `
            <div class="welcome-placeholder">
                <span class="placeholder-icon">💡</span>
                <p>Paste a coding problem and click Solve. Qwen will generate brute force and optimized solutions, and Study Buddy will teach them in deep, comprehensive detail here.</p>
            </div>
        `;
        return;
    }

    exportBtn.style.display = "inline-flex";
    exportBtn.href = `/session/${sessionId}/export?mode=leetcode`;

    history.forEach(msg => {
        const titleDiv = document.createElement("h2");
        titleDiv.className = "history-question-title";
        titleDiv.textContent = `🔍 Code Challenge: ${msg.question.substring(0, 45)}...`;
        viewport.appendChild(titleDiv);

        const cardDiv = document.createElement("div");
        cardDiv.className = "history-card-body";
        
        const mdText = msg.card.text || "";
        cardDiv.innerHTML = parseMarkdown(mdText);

        viewport.appendChild(cardDiv);
    });

    setTimeout(() => {
        viewport.scrollTop = viewport.scrollHeight;
    }, 50);
}

// 7. Settings & Ingestion Operations
async function triggerIngestion(slug) {
    const btn = document.getElementById(`btn-ingest-${slug}`);
    btn.disabled = true;
    btn.textContent = "Working...";
    
    const container = document.getElementById(`progress-container-${slug}`);
    container.style.display = "block";

    try {
        await apiCall(`/subjects/${slug}/ingest`, "POST");
        pollIngestStatus(slug);
    } catch (e) {
        showGlobalError(`Failed to trigger ingestion for ${slug}.`);
        btn.disabled = false;
        btn.textContent = "Index";
        container.style.display = "none";
    }
}

function pollIngestStatus(slug) {
    const interval = setInterval(async () => {
        try {
            const job = await apiCall(`/subjects/${slug}/ingest/status`);
            const bar = document.getElementById(`progress-bar-${slug}`);
            const cell = document.getElementById(`ingest-status-${slug}`);
            const btn = document.getElementById(`btn-ingest-${slug}`);
            const container = document.getElementById(`progress-container-${slug}`);
            
            if (job.state === "running") {
                cell.innerHTML = `<span class="badge" style="background-color: rgba(139, 92, 246, 0.15); color: var(--accent-hover)">Indexing...</span>`;
                
                const p = job.progress || {};
                let pct = 5;
                if (p.phase === "scanning") pct = 10;
                else if (p.phase === "embedding" && p.total_chunks > 0) {
                    pct = 15 + Math.min(80, (p.chunk_index / p.total_chunks) * 80);
                } else if (p.phase === "topics") pct = 95;
                else if (p.phase === "topics_merge") pct = 98;
                
                bar.style.width = `${pct}%`;
            } else if (job.state === "done") {
                clearInterval(interval);
                bar.style.width = "100%";
                cell.innerHTML = `<span class="badge" style="background-color: rgba(16, 185, 129, 0.15); color: var(--color-success)">Indexed</span>`;
                btn.disabled = false;
                btn.textContent = "Re-Index";
                setTimeout(() => { container.style.display = "none"; }, 1500);
                
                await loadSubjects();
                if (activeSubjectSlug === slug) {
                    selectSubject(slug, activeSubjectName);
                }
            } else if (job.state === "error") {
                clearInterval(interval);
                cell.innerHTML = `<span class="badge" style="background-color: rgba(239, 68, 68, 0.15); color: var(--color-error)">Error</span>`;
                btn.disabled = false;
                btn.textContent = "Index";
                container.style.display = "none";
                showGlobalError(`Ingestion job for ${slug} failed: ${job.error}`);
            }
        } catch (e) {
            clearInterval(interval);
        }
    }, 2000);
}

// 8. Event Binding & Utilities
function setupEventListeners() {
    document.getElementById("btn-leetcode-solve").addEventListener("click", solveLeetCode);

    document.getElementById("btn-subjects-followup-send").addEventListener("click", sendSubjectFollowup);
    document.getElementById("subjects-followup-input").addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            sendSubjectFollowup();
        }
    });
    
    window.triggerIngestion = triggerIngestion;
}

function setAskingState(asking, type) {
    isAsking = asking;
    const expStatus = document.getElementById("explanation-status");
    const lcStatus = document.getElementById("leetcode-status-msg");
    const lcBtn = document.getElementById("btn-leetcode-solve");
    const followupInput = document.getElementById("subjects-followup-input");
    const followupBtn = document.getElementById("btn-subjects-followup-send");

    if (asking) {
        if (type === "explanation") {
            expStatus.textContent = "Thinking...";
            if (followupInput) followupInput.disabled = true;
            if (followupBtn) followupBtn.disabled = true;
        } else if (type === "leetcode") {
            lcStatus.textContent = "Thinking...";
            lcBtn.disabled = true;
        }
    } else {
        expStatus.textContent = "Ready";
        lcStatus.textContent = "";
        lcBtn.disabled = false;
        if (followupInput) {
            followupInput.disabled = false;
            followupInput.focus();
        }
        if (followupBtn) followupBtn.disabled = false;
    }
}

function showGlobalError(msg) {
    console.error(msg);
    alert(msg);
}

function parseMarkdown(mdText) {
    if (!mdText) return "";
    
    let html = mdText
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
        
    let parts = html.split("```");
    for (let i = 1; i < parts.length; i += 2) {
        let code = parts[i];
        let lines = code.split("\n");
        let firstLine = lines[0].trim();
        if (["python", "javascript", "js", "py", "bash", "html", "css", "sql", "c++", "cpp"].includes(firstLine.toLowerCase())) {
            lines.shift();
        }
        let cleanCode = lines.join("\n").trim();
        parts[i] = `<div class="code-container"><button class="copy-btn" onclick="copyCode(this)">Copy</button><pre><code>${cleanCode}</code></pre></div>`;
    }
    html = parts.join("");
    
    let lines = html.split("\n");
    let inCode = false;
    for (let i = 0; i < lines.length; i++) {
        let line = lines[i];
        
        if (line.includes("<div class=\"code-container\">")) {
            inCode = true;
        }
        if (line.includes("</div>") && inCode) {
            inCode = false;
            continue;
        }
        if (inCode) continue;
        
        if (line.startsWith("# ")) {
            lines[i] = `<h1>${line.substring(2)}</h1>`;
        } else if (line.startsWith("## ")) {
            lines[i] = `<h2>${line.substring(3)}</h2>`;
        } else if (line.startsWith("### ")) {
            lines[i] = `<h3>${line.substring(4)}</h3>`;
        } else if (line.startsWith("- ") || line.startsWith("* ")) {
            lines[i] = `<li>${line.substring(2)}</li>`;
        } else if (line.trim() !== "") {
            lines[i] = `<p>${line}</p>`;
        }
        
        lines[i] = lines[i]
            .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
            .replace(/`(.*?)`/g, "<code>$1</code>");
    }
    
    return lines.join("\n");
}

function copyCode(btn) {
    const code = btn.nextElementSibling.querySelector("code").textContent;
    navigator.clipboard.writeText(code).then(() => {
        btn.textContent = "Copied!";
        btn.style.backgroundColor = "var(--color-success)";
        btn.style.borderColor = "var(--color-success)";
        setTimeout(() => {
            btn.textContent = "Copy";
            btn.style.backgroundColor = "rgba(255, 255, 255, 0.05)";
            btn.style.borderColor = "rgba(255, 255, 255, 0.1)";
        }, 1500);
    });
}
window.copyCode = copyCode;
