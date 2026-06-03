"""
Simple HTTP server to host the search UI.
Serves the search_ui.html file with proper CORS headers.
"""
import http.server
import socketserver
import webbrowser
import os

PORT = 8080

class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler with CORS support."""
    
    def end_headers(self):
        """Add CORS headers to all responses."""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS preflight."""
        self.send_response(200)
        self.end_headers()
    
    def log_message(self, format, *args):
        """Custom log format."""
        if '200' in str(args[0]):
            print(f"✓ {args[0]}")
        else:
            print(f"  {args[0]}")

if __name__ == "__main__":
    # Change to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("="*80)
    print("MAR VISTA LIBRARIAN - SEARCH UI SERVER")
    print("="*80)
    print(f"\n🌐 Starting UI server on http://localhost:{PORT}")
    print(f"📁 Serving from: {script_dir}")
    print(f"\n📖 Search Interface: http://localhost:{PORT}/search_ui.html")
    print(f"🔌 API Server (must be running): http://localhost:8000")
    print("\nPress CTRL+C to stop the server")
    print("="*80 + "\n")
    
    # Verify file exists
    if not os.path.exists('search_ui.html'):
        print("❌ Error: search_ui.html not found in current directory!")
        exit(1)
    
    # Create server
    with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
        # Open browser automatically
        url = f"http://localhost:{PORT}/search_ui.html"
        print(f"🚀 Opening browser at {url}...\n")
        webbrowser.open(url)
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n✓ UI server stopped")
