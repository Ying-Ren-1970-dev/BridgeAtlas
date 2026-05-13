"""Interactive CLI for the Librarian system."""
import cmd
from typing import Optional

from main import LibrarianApp


class LibrarianCLI(cmd.Cmd):
    """Interactive command-line interface for Librarian."""
    
    intro = """
╔════════════════════════════════════════════════════════════════╗
║                    LIBRARIAN v1.0                              ║
║          Structural Engineering Project Search System          ║
╚════════════════════════════════════════════════════════════════╝

Type 'help' to see available commands.
Type 'exit' or 'quit' to leave the program.
"""
    prompt = '\nLibrarian> '
    
    def __init__(self):
        """Initialize the CLI."""
        super().__init__()
        self.app = LibrarianApp()
    
    def do_build(self, arg):
        """
        Build or rebuild the knowledge base from PDF files.
        Usage: build [--clear]
        
        Options:
            --clear    Clear existing data before building
        """
        clear = '--clear' in arg
        self.app.build_knowledge_base(clear_existing=clear)
    
    def do_search(self, arg):
        """
        Search the knowledge base.
        Usage: search <query>
        
        Example: search retaining wall foundation design
        """
        if not arg:
            print("Error: Please provide a search query")
            return
        
        results = self.app.search(arg)
        self.app.display_search_results(results)
    
    def do_filter(self, arg):
        """
        Advanced search with filters.
        Usage: filter <query> --category=<cat> --phase=<phase> --engineer=<name>
        
        Example: filter bridge design --category=Bridges --phase=100% Final
        """
        # Parse arguments
        parts = arg.split('--')
        query = parts[0].strip()
        
        filters = {}
        for part in parts[1:]:
            if '=' in part:
                key, value = part.split('=', 1)
                filters[key.strip()] = value.strip()
        
        if not query:
            print("Error: Please provide a search query")
            return
        
        results = self.app.search(query, filters=filters)
        self.app.display_search_results(results)
    
    def do_list(self, arg):
        """
        List all indexed projects.
        Usage: list
        """
        self.app.list_all_projects()
    
    def do_stats(self, arg):
        """
        Show knowledge base statistics.
        Usage: stats
        """
        self.app.show_statistics()
    
    def do_details(self, arg):
        """
        Show detailed information about a project.
        Usage: details <file_name>
        
        Example: details 30%_Elk Grove Station Structure Plan Set.pdf
        """
        if not arg:
            print("Error: Please provide a file name")
            return
        
        self.app.get_project_details(arg)
    
    def do_categories(self, arg):
        """
        List all categories.
        Usage: categories
        """
        import config
        print("\nAvailable Categories:")
        for i, cat in enumerate(config.STRUCTURAL_CATEGORIES, 1):
            print(f"  {i}. {cat}")
    
    def do_phases(self, arg):
        """
        List all plan phases.
        Usage: phases
        """
        import config
        print("\nPlan Phases:")
        for i, phase in enumerate(config.PLAN_PHASES, 1):
            print(f"  {i}. {phase}")
    
    def do_clear(self, arg):
        """
        Clear the screen.
        Usage: clear
        """
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
        print(self.intro)
    
    def do_exit(self, arg):
        """
        Exit the program.
        Usage: exit
        """
        print("\nGoodbye!")
        return True
    
    def do_quit(self, arg):
        """
        Exit the program.
        Usage: quit
        """
        return self.do_exit(arg)
    
    def do_help(self, arg):
        """
        Show help information.
        Usage: help [command]
        """
        if arg:
            super().do_help(arg)
        else:
            print("\n" + "=" * 70)
            print("AVAILABLE COMMANDS")
            print("=" * 70)
            print("\nKnowledge Base Management:")
            print("  build [--clear]       - Build knowledge base from PDFs")
            print("  stats                 - Show knowledge base statistics")
            print("\nSearch Commands:")
            print("  search <query>        - Search for projects")
            print("  filter <query> --key=val - Advanced search with filters")
            print("\nProject Information:")
            print("  list                  - List all indexed projects")
            print("  details <file>        - Show project details")
            print("\nReference:")
            print("  categories            - List available categories")
            print("  phases                - List plan phases")
            print("\nUtility:")
            print("  clear                 - Clear the screen")
            print("  help [command]        - Show help")
            print("  exit, quit            - Exit the program")
            print("\nExample Searches:")
            print("  search retaining wall seismic design")
            print("  search tunnel boring foundation")
            print("  filter bridge --category=Bridges --phase=Final")
    
    def emptyline(self):
        """Do nothing on empty line."""
        pass
    
    def default(self, line):
        """Handle unknown commands."""
        print(f"Unknown command: '{line}'. Type 'help' for available commands.")


def main():
    """Run the interactive CLI."""
    try:
        LibrarianCLI().cmdloop()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
    except Exception as e:
        print(f"\nError: {str(e)}")


if __name__ == '__main__':
    main()
