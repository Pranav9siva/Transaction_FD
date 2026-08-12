import re

def update_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace root CSS variables
    root_pattern = re.compile(r':root\s*\{.*?\}(?=\s*\* \{)', re.DOTALL)
    new_root = """
        :root {
            --c-morning-sky: #87ceeb;
            --c-honey: #ffc30b;
            --c-cerulean: #007ba7;
            --c-mist: #bcc6cc;

            /* Dark Theme Default */
            --bg-main: #00151f;
            --card-bg: rgba(0, 40, 60, 0.7);
            --card-border: rgba(188, 198, 204, 0.15);
            --text-main: var(--c-morning-sky);
            --text-muted: var(--c-mist);
            
            --accent-primary: var(--c-cerulean);
            --accent-secondary: var(--c-morning-sky);
            --accent-warning: var(--c-honey);
            
            --success-color: var(--c-morning-sky);
            --warning-color: var(--c-honey);
            --danger-color: var(--c-honey);
        }

        [data-theme="light"] {
            --bg-main: #eef2f5;
            --card-bg: rgba(255, 255, 255, 0.85);
            --card-border: rgba(188, 198, 204, 0.6);
            --text-main: var(--c-cerulean);
            --text-muted: #5a7684;
            --success-color: var(--c-cerulean);
            --warning-color: var(--c-honey);
            --danger-color: var(--c-honey);
        }
"""
    content = root_pattern.sub(new_root.strip(), content)

    # Replace variables usage
    replacements = {
        'var(--success-green)': 'var(--success-color)',
        'var(--warning-orange)': 'var(--warning-color)',
        'var(--danger-red)': 'var(--danger-color)',
        'var(--accent-cyan)': 'var(--accent-secondary)',
        'var(--accent-purple)': 'var(--accent-primary)',
        'var(--accent-blue)': 'var(--accent-primary)',
        'var(--bg-dark)': 'var(--bg-main)',
        '#fff': 'var(--text-main)',
        '#ffffff': 'var(--text-main)',
        'rgba(255, 255, 255,': 'rgba(188, 198, 204,',
        'rgba(16, 172, 132,': 'rgba(135, 206, 235,',
        'rgba(255, 159, 67,': 'rgba(255, 195, 11,',
        'rgba(255, 75, 92,': 'rgba(255, 195, 11,',
        'rgba(79, 172, 254,': 'rgba(0, 123, 167,',
        'rgba(127, 0, 255,': 'rgba(135, 206, 235,'
    }
    for old, new in replacements.items():
        content = content.replace(old, new)
        
    # Update chart colors in JS
    content = content.replace("'#00f2fe'", "'#87ceeb'")
    content = content.replace("rgba(0, 242, 254, 0.1)", "rgba(135, 206, 235, 0.1)")
    content = content.replace("'#4facfe'", "'#007ba7'")
    content = content.replace("'#7f00ff'", "'#bcc6cc'")
    content = content.replace("'#10ac84'", "'#ffc30b'")

    # Add toggle button to header
    header_pattern = re.compile(r'<div class="status-badge">')
    toggle_html = """
            <button class="nav-btn" onclick="toggleTheme()" style="margin-right: 1rem; border: 1px solid var(--card-border);">Toggle Theme</button>
            <div class="status-badge">
"""
    content = content.replace('<div class="status-badge">', toggle_html.strip(), 1)

    # Add toggle script to the end of <script>
    script_pattern = re.compile(r'</script>\s*</body>')
    toggle_script = """
        function toggleTheme() {
            const currentTheme = document.documentElement.getAttribute('data-theme');
            if (currentTheme === 'light') {
                document.documentElement.setAttribute('data-theme', 'dark');
            } else {
                document.documentElement.setAttribute('data-theme', 'light');
            }
        }
    </script>
</body>"""
    content = script_pattern.sub(toggle_script.strip(), content)

    # Update body tag to have transition
    content = content.replace('body {', 'body {\n            transition: background-color 0.3s ease, color 0.3s ease;')
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

update_html('c:\\Desktop\\soft_eng\\templates\\index.html')
print("Successfully updated HTML and injected colors + theme toggle.")
