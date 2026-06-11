"""Main application module for the Librarian system."""
from pathlib import Path
from typing import Optional
import json

import config
from pdf_processor import PDFProcessor
from vector_store import VectorStore
from categorizer import StructuralCategorizer
from metadata_manager import MetadataManager
from search_agent import SearchAgent
from page_classifier import PageClassifier
from enriched_metadata_graph_builder import GraphIntegrationManager
from classification_training import SearchClassificationTrainer
from search_feedback_learner import learn_from_feedback
from feedback_cleanup import purge_legacy_feedback
from project_profile_builder import ProjectProfileBuilder
from storage_adapter import cloud_sync_configured, storage, storage_for_cloud_sync


class LibrarianApp:
    """Main application class for the Librarian system."""
    
    def __init__(self):
        """Initialize the Librarian application."""
        self.pdf_processor = PDFProcessor()
        self.vector_store = VectorStore()
        self.categorizer = StructuralCategorizer()
        self.metadata_manager = MetadataManager()
        self.search_agent = SearchAgent()
        self.graph_manager = GraphIntegrationManager()
    
    def _is_excluded_project_path(self, pdf_path):
        normalized_path = str(pdf_path).replace('\\', '/').lower()
        return any(pattern in normalized_path for pattern in config.PROJECTS_FOLDER_EXCLUDE)

    def build_knowledge_base(
        self,
        projects_folder: Optional[Path] = None,
        clear_existing: bool = False
    ):
        """
        Build the knowledge base from PDF files.
        Saves progress after each file to avoid data loss.
        
        Args:
            projects_folder: Path to projects folder (defaults to config)
            clear_existing: Whether to clear existing data before building
        """
        from tqdm import tqdm
        
        projects_folder = projects_folder or config.PROJECTS_FOLDER
        
        print("=" * 60)
        print("LIBRARIAN - Building Knowledge Base")
        print("=" * 60)
        print("(Saving progress after each file)")
        
        if clear_existing:
            print("\nClearing existing data...")
            self.vector_store.clear_collection()
        
        # Initialize vector store
        self.vector_store.initialize_vectorstore()
        
        # Get all PDF files, excluding configured ignore directories
        all_pdfs = list(projects_folder.rglob('*.pdf'))
        pdf_files = [pdf_path for pdf_path in all_pdfs if not self._is_excluded_project_path(pdf_path)]
        excluded_count = len(all_pdfs) - len(pdf_files)

        print(f"\nFound {len(pdf_files)} PDF files")
        if excluded_count:
            print(f"Excluded {excluded_count} PDF files from ignored folders: {', '.join(config.PROJECTS_FOLDER_EXCLUDE)}")
        
        if not pdf_files:
            print("No PDF files found. Exiting.")
            return
        
        # Process each file incrementally
        total_chunks = 0
        processed_count = 0
        
        print("\nProcessing files (saving after each)...")
        for pdf_path in tqdm(pdf_files, desc="Building Knowledge Base"):
            try:
                # Step 1: Process PDF
                pdf_data = self.pdf_processor.process_pdf(pdf_path)
                if not pdf_data:
                    continue
                
                # Step 2: Categorize project
                # Extract text sample from chunks for categorization
                text_sample = " ".join([chunk['text'] for chunk in pdf_data['chunks'][:5]])[:3000]
                # Extract clean project name from filename
                project_name = self.pdf_processor._extract_project_name(pdf_data['file_name'])
                categories = self.categorizer.categorize_project(
                    project_name=project_name,
                    text_sample=text_sample,
                    metadata=pdf_data['metadata']
                )
                
                # Step 3: Add to vector store (persists automatically)
                chunks_added = self.vector_store.add_documents([pdf_data])
                total_chunks += chunks_added
                
                # Step 4: Save metadata immediately
                self.metadata_manager.add_project_metadata(
                    file_name=pdf_data['file_name'],
                    file_path=pdf_data['file_path'],
                    metadata=pdf_data['metadata'],
                    categories=categories,
                    total_pages=pdf_data['total_pages']
                )
                self.metadata_manager.save()
                
                processed_count += 1
                
            except Exception as e:
                print(f"\nError processing {pdf_path.name}: {str(e)}")
                print("Continuing with next file...")
                continue
        
        # Display summary
        print("\n" + "=" * 60)
        print("KNOWLEDGE BASE BUILD COMPLETE")
        print("=" * 60)
        print(f"Total Projects: {processed_count}")
        print(f"Total Chunks: {total_chunks}")
        print(f"Vector DB Path: {config.VECTOR_DB_PATH}")
        print(f"Metadata DB Path: {config.METADATA_DB_PATH}")
        
        # Show statistics
        if processed_count > 0:
            stats = self.metadata_manager.get_stats()
            print(f"\nCategory Distribution:")
            for category, count in sorted(stats['category_distribution'].items()):
                print(f"  {category}: {count}")
            
            print(f"\nPhase Distribution:")
            for phase, count in sorted(stats['phase_distribution'].items()):
                print(f"  {phase}: {count}")
    
    def update_file(self, file_path: Path):
        """Update a single file in the knowledge base."""
        from tqdm import tqdm
        
        print(f"\n{'='*60}")
        print(f"Updating: {file_path.name}")
        print(f"{'='*60}")
        
        # Initialize vector store
        self.vector_store.initialize_vectorstore()
        
        # Delete old data for this file
        self.vector_store.delete_by_filename(file_path.name)
        self.metadata_manager.delete_project(file_path.name)
        
        try:
            # Process PDF
            pdf_data = self.pdf_processor.process_pdf(file_path)
            if not pdf_data:
                print(f"Failed to process {file_path.name}")
                return
            
            # Categorize project
            text_sample = " ".join([chunk['text'] for chunk in pdf_data['chunks'][:5]])[:3000]
            project_name = self.pdf_processor._extract_project_name(pdf_data['file_name'])
            categories = self.categorizer.categorize_project(
                project_name=project_name,
                text_sample=text_sample,
                metadata=pdf_data['metadata']
            )
            
            # Add to vector store
            chunks_added = self.vector_store.add_documents([pdf_data])
            
            # Save metadata
            self.metadata_manager.add_project_metadata(
                file_name=pdf_data['file_name'],
                file_path=pdf_data['file_path'],
                metadata=pdf_data['metadata'],
                categories=categories,
                total_pages=pdf_data['total_pages']
            )
            self.metadata_manager.save()
            
            print(f"\n✓ Successfully updated {file_path.name}")
            print(f"  Total Pages: {pdf_data['total_pages']}")
            print(f"  Pages with Text: {pdf_data.get('pages_with_text', 'N/A')}")
            print(f"  Chunks: {chunks_added}")
            print(f"  Categories: {', '.join(categories)}")
            
        except Exception as e:
            print(f"\nError updating {file_path.name}: {str(e)}")
            raise

    def clean_excluded_projects(self) -> None:
        """Remove indexed projects that live in excluded folders."""
        excluded_patterns = config.PROJECTS_FOLDER_EXCLUDE
        if not excluded_patterns:
            print("No excluded project filters configured.")
            return

        print("\nCleaning excluded project entries from the vector store and metadata...")
        self.vector_store.initialize_vectorstore()

        projects = self.metadata_manager.get_all_projects()
        excluded_files = [
            file_name
            for file_name, project in projects.items()
            if any(pattern in str(project.get('file_path', '')).replace('\\', '/').lower() for pattern in excluded_patterns)
        ]

        if not excluded_files:
            print("No excluded projects found in metadata.")
            return

        deleted_vectors = self.vector_store.delete_by_file_names(excluded_files)
        deleted_metadata = self.metadata_manager.delete_projects_by_path_filters(excluded_patterns)
        self.metadata_manager.save()

        print(f"\n✓ Removed {deleted_metadata} excluded project metadata entries")
        print(f"✓ Removed {deleted_vectors} excluded project vector entries")

    def sync_cloud(self):
        """Sync the current local vector store and metadata to cloud storage."""
        if not cloud_sync_configured():
            print("\nCloud sync is not configured.")
            print("Set these in .env (USE_CLOUD_STORAGE can stay false locally):")
            print("  GCS_BUCKET_PDFS=<your-pdf-bucket>")
            print("  GCS_BUCKET_VECTORS=<your-vector-bucket>")
            print("  GCP_PROJECT_ID=<your-gcp-project-id>")
            return

        print("\nSyncing local knowledge base to cloud storage...")
        if not storage.use_gcs:
            print("  (Enabling cloud sync for this command; local USE_CLOUD_STORAGE=false is OK)")
        cloud_storage = storage_for_cloud_sync()
        cloud_storage.sync_vector_db_to_cloud()

    def search(
        self,
        query: str,
        max_results: int = 10,
        filters: Optional[dict] = None
    ) -> dict:
        """
        Search the knowledge base.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            filters: Optional filters
            
        Returns:
            Search results dictionary
        """
        if filters:
            results = self.search_agent.advanced_search(query, filters)
        else:
            results = self.search_agent.search(query, k=max_results)
        
        return results
    
    def display_search_results(self, results: dict):
        """
        Display search results in a formatted manner.
        
        Args:
            results: Search results from search_agent
        """
        print("\n" + "=" * 80)
        print(f"SEARCH RESULTS FOR: '{results['query']}'")
        print("=" * 80)
        
        print(f"\n{results['summary']}\n")
        
        if not results['results']:
            print("No results found.")
            return
        
        print(f"\nFound {results['total_projects']} relevant project(s):\n")
        
        for i, result in enumerate(results['results'], 1):
            print(f"\n{'-' * 80}")
            print(f"[{i}] PROJECT: {result['project_name']}")
            print(f"{'-' * 80}")
            print(f"File Name:          {result['file_name']}")
            print(f"Phase:              {result['phase']}")
            print(f"Engineer of Record: {result['engineer_of_record']}")
            print(f"Date:               {result['date']}")
            print(f"Categories:         {', '.join(result['categories'])}")
            print(f"Relevant Pages:     {', '.join(map(str, result['relevant_pages'][:20]))}")
            if len(result['relevant_pages']) > 20:
                print(f"                    ... and {len(result['relevant_pages']) - 20} more pages")
            print(f"Total Pages:        {result['total_pages']}")
            print(f"\nSample Content:")
            print(f"  {result['sample_content']}")
    
    def list_all_projects(self):
        """List all indexed projects."""
        projects = self.search_agent.list_all_projects()
        
        print("\n" + "=" * 80)
        print("ALL INDEXED PROJECTS")
        print("=" * 80)
        print(f"\nTotal: {len(projects)} projects\n")
        
        for i, project in enumerate(projects, 1):
            print(f"{i}. {project['project_name']}")
            print(f"   File: {project['file_name']}")
            print(f"   Phase: {project['phase']}")
            print(f"   Categories: {', '.join(project['categories'])}")
            print(f"   Pages: {project['total_pages']}")
            print()
    
    def show_statistics(self):
        """Display knowledge base statistics."""
        stats = self.search_agent.get_statistics()
        
        print("\n" + "=" * 80)
        print("KNOWLEDGE BASE STATISTICS")
        print("=" * 80)
        
        print("\nMetadata Database:")
        print(f"  Total Projects: {stats['metadata']['total_projects']}")
        print(f"  Last Updated: {stats['metadata']['last_updated']}")
        
        print("\n  Category Distribution:")
        for category, count in sorted(stats['metadata']['category_distribution'].items()):
            print(f"    {category}: {count}")
        
        print("\n  Phase Distribution:")
        for phase, count in sorted(stats['metadata']['phase_distribution'].items()):
            print(f"    {phase}: {count}")
        
        print("\nVector Store:")
        print(f"  Collection: {stats['vector_store']['collection_name']}")
        print(f"  Document Chunks: {stats['vector_store']['document_count']}")
        print(f"  Path: {stats['vector_store']['path']}")
    
    def get_project_details(self, file_name: str):
        """
        Get and display detailed information about a project.
        
        Args:
            file_name: Name of the project file
        """
        project = self.search_agent.get_project_details(file_name)
        
        if not project:
            print(f"\nProject '{file_name}' not found.")
            return
        
        print("\n" + "=" * 80)
        print(f"PROJECT DETAILS: {project['project_name']}")
        print("=" * 80)
        print(f"\nFile Name:          {project['file_name']}")
        print(f"File Path:          {project['file_path']}")
        print(f"Project Name:       {project['project_name']}")
        print(f"Phase:              {project['phase']}")
        print(f"Engineer of Record: {project['engineer_of_record']}")
        print(f"Date:               {project['date']}")
        print(f"Categories:         {', '.join(project['categories'])}")
        print(f"Total Pages:        {project['total_pages']}")
        print(f"Indexed At:         {project['indexed_at']}")
    
    def enrich_metadata(self, file_path: Path, sample_pages: Optional[list] = None):
        """
        Enrich knowledge base with page classification and title block extraction.
        
        Args:
            file_path: Path to PDF file to enrich
            sample_pages: Optional list of specific pages to analyze (1-indexed)
        """
        print(f"\n{'='*60}")
        print(f"Enriching Metadata: {file_path.name}")
        print(f"{'='*60}")
        
        classifier = PageClassifier()
        
        # Classify pages and extract title blocks
        results = classifier.batch_classify_pdf(file_path, sample_pages)
        
        if not results:
            print("No results obtained from classification.")
            return
        
        # Display results
        print(f"\n✓ Analyzed {len(results)} pages")
        print(f"\n{'='*60}")
        print("CLASSIFICATION SUMMARY")
        print(f"{'='*60}")
        
        plan_count = sum(1 for r in results if r.get('page_type') == 'plan')
        report_count = sum(1 for r in results if r.get('page_type') == 'report')
        unknown_count = sum(1 for r in results if r.get('page_type') == 'unknown')
        
        print(f"\nPlan/CAD Pages:   {plan_count}")
        print(f"Report Pages:     {report_count}")
        print(f"Unknown:          {unknown_count}")
        
        # Display detailed results for plans
        plans = [r for r in results if r.get('page_type') == 'plan']
        if plans:
            print(f"\n{'='*60}")
            print("PLAN PAGES WITH TITLE BLOCKS")
            print(f"{'='*60}")
            
            for plan in plans:
                print(f"\n--- Page {plan['page_number']} ---")
                print(f"Confidence: {plan.get('confidence', 0):.2f}")
                
                tb = plan.get('title_block')
                if tb:
                    # Project Info
                    proj_info = tb.get('project_info', {})
                    if proj_info and any(v for v in proj_info.values() if v):
                        print("\nProject Information:")
                        for key, value in proj_info.items():
                            if value:
                                print(f"  {key.replace('_', ' ').title()}: {value}")
                    
                    # Plan Type
                    plan_type = tb.get('plan_type', {})
                    if plan_type and any(v for v in plan_type.values() if v):
                        print("\nPlan Type:")
                        for key, value in plan_type.items():
                            if value:
                                print(f"  {key.replace('_', ' ').title()}: {value}")
                    
                    # Plan Contents
                    contents = tb.get('plan_contents', {})
                    if contents:
                        print("\nPlan Contents:")
                        if contents.get('has_tables'):
                            print("  ✓ Contains tables")
                        if contents.get('has_diagrams'):
                            print("  ✓ Contains diagrams")
                        if contents.get('structural_elements'):
                            print(f"  Elements: {', '.join(contents['structural_elements'][:10])}")
                        if contents.get('detail_types'):
                            print(f"  Detail Types: {', '.join(contents['detail_types'])}")
                        if contents.get('grid_references'):
                            print(f"  Grid References: {', '.join(contents['grid_references'])}")
        
        # Save enriched metadata
        config.DATA_FOLDER.mkdir(exist_ok=True)
        enriched_file = config.DATA_FOLDER / f"enriched_{file_path.stem}.json"
        try:
            with open(enriched_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Enriched metadata saved to: {enriched_file}")
        except Exception as e:
            print(f"\nError saving enriched metadata: {str(e)}")
    
    def enrich_knowledge_base(self, projects_folder: Optional[Path] = None):
        """
        Add enrichment metadata to existing knowledge base without re-processing PDFs.
        Preserves existing Vision analysis while adding title block and page classification data.
        No API calls required - loads from existing enriched JSON files.
        
        Args:
            projects_folder: Path to projects folder (defaults to config)
        """
        projects_folder = projects_folder or config.PROJECTS_FOLDER
        
        print("=" * 60)
        print("ENRICHING EXISTING KNOWLEDGE BASE")
        print("=" * 60)
        print("(Preserving existing Vision analysis)")
        print("(Loading enrichment from JSON files)")
        
        # Initialize vector store
        self.vector_store.initialize_vectorstore()
        
        # Find all enriched JSON files
        enriched_files = list(config.DATA_FOLDER.glob('enriched_*.json'))
        
        if not enriched_files:
            print("\n❌ No enriched metadata files found!")
            print("   Run enrichment first: python main.py enrich <file>")
            return
        
        print(f"\nFound {len(enriched_files)} enriched metadata files")
        
        total_updated = 0
        success_count = 0
        
        for enriched_file in enriched_files:
            # Extract original filename from enriched filename
            # Format: enriched_<filename>.json -> <filename>.pdf
            original_name = enriched_file.stem.replace('enriched_', '') + '.pdf'
            
            print(f"\nProcessing: {original_name}")
            
            try:
                updated = self.vector_store.enrich_existing_documents(original_name)
                total_updated += updated
                if updated > 0:
                    success_count += 1
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
                continue
        
        print("\n" + "=" * 60)
        print("ENRICHMENT COMPLETE")
        print("=" * 60)
        print(f"✓ Processed {len(enriched_files)} files")
        print(f"✓ Successfully enriched {success_count} files")
        print(f"✓ Total chunks updated: {total_updated}")
        print(f"✓ Cost: $0 (no API calls)")
    
    def build_detail_graphs(self):
        """
        Build detail graphs from enriched metadata.
        Creates connectivity graphs for each project to enable detailed search and navigation.
        
        Uses the Algorithmic Graph Approach:
        - Deterministic parsing of PDF structure (no ML)
        - Graph storage of detail relationships
        - Grounds AI queries with structural connectivity
        """
        print("=" * 60)
        print("BUILDING DETAIL GRAPHS")
        print("=" * 60)
        print("(Creating connectivity graphs from enriched metadata)")
        
        # Find all enriched JSON files
        enriched_files = list(config.DATA_FOLDER.glob('enriched_*.json'))
        
        if not enriched_files:
            print("\n❌ No enriched metadata files found!")
            print("   Run enrichment first: python main.py enrich <file>")
            return
        
        print(f"\nFound {len(enriched_files)} enriched metadata files")
        
        from enriched_metadata_loader import EnrichedMetadataLoader
        loader = EnrichedMetadataLoader()
        
        graphs_built = 0
        
        for enriched_file in enriched_files:
            # Extract original filename from enriched filename
            # Format: enriched_<filename>.json -> <filename>.pdf
            project_name = enriched_file.stem.replace('enriched_', '')
            pdf_file_name = f"{project_name}.pdf"
            
            print(f"\nBuilding graph for: {project_name}")
            
            try:
                # Load enriched metadata
                enriched_metadata = loader.load_enriched_metadata(pdf_file_name)
                
                if not enriched_metadata:
                    print(f"  ❌ Failed to load enriched metadata")
                    continue
                
                # Build and store graph
                graph = self.graph_manager.build_and_store_graph(
                    enriched_metadata=enriched_metadata,
                    pdf_file_name=pdf_file_name,
                    project_name=project_name
                )
                
                print(f"  ✓ Graph built: {len(graph.nodes)} details, "
                      f"{sum(len(e) for e in graph.edges.values())} relationships")
                graphs_built += 1
                
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        print("\n" + "=" * 60)
        print("DETAIL GRAPHS COMPLETE")
        print("=" * 60)
        print(f"✓ Built {graphs_built}/{len(enriched_files)} graphs")
        print(f"✓ Graphs stored in: {self.graph_manager.data_dir}")

    def index_detail_nodes(self):
        """
        Index detail nodes from graphs into the vector store for cross-project search.
        
        Creates separate embeddings for each structural detail (section view, elevation, etc.)
        enabling fine-grained search and discovery across all projects.
        """
        print("=" * 60)
        print("INDEXING DETAIL NODES")
        print("=" * 60)
        print("(Creating embeddings for individual structural details)")
        
        from detail_indexer import DetailIndexer
        
        indexer = DetailIndexer()
        
        if not indexer.initialize():
            print("\n❌ Failed to initialize detail indexer")
            return
        
        print("\nClearing previous detail index...")
        indexer.clear_collection()
        
        print("Indexing details from all graphs...")
        total_indexed = indexer.index_all_graphs()
        
        print("\n" + "=" * 60)
        print("DETAIL INDEXING COMPLETE")
        print("=" * 60)
        print(f"✓ Indexed {total_indexed} detail nodes")
        print(f"✓ Details stored in: {indexer.vector_db_path}")
        print("\nDetail-level search is now available!")

    def build_enhanced_topology(self):
        """
        Build enhanced topology JSON files for all enriched metadata projects.

        Uses existing enriched_*.json files in data/ (no API calls).
        Outputs data/enhanced_topology_<project>.json plus a manifest.
        """
        from enhanced_topology_builder import EnhancedTopologyBuilder

        builder = EnhancedTopologyBuilder()
        builder.build_all()

    def merge_deep_vision_topology(self):
        """
        Merge deep-vision enhanced topology into the vector knowledge base.

        Loads data/enhanced_topology_*.json files and appends per-page topology
        to existing chunks. Preserves PDF text and enriched metadata.
        Re-embeds only chunks that receive new topology text.
        """
        print("=" * 60)
        print("MERGING DEEP VISION TOPOLOGY INTO KNOWLEDGE BASE")
        print("=" * 60)
        print("(Preserving existing PDF text and enriched metadata)")

        self.vector_store.initialize_vectorstore()

        from enhanced_topology_loader import EnhancedTopologyLoader

        topology_files = EnhancedTopologyLoader.list_deep_vision_topology_files()
        if not topology_files:
            print("\nNo deep vision topology files found in data/")
            print("Run deep vision first: python topology_analyzer.py --all")
            return

        print(f"\nFound {len(topology_files)} deep vision topology files")
        print("(JSON source files are read-only and never modified)")

        total_updated = 0
        success_count = 0

        for topology_file in topology_files:
            file_name = topology_file.stem.replace("enhanced_topology_", "") + ".pdf"
            print(f"\nProcessing: {file_name}")
            try:
                updated = self.vector_store.merge_deep_vision_topology(file_name)
                total_updated += updated
                if updated > 0:
                    success_count += 1
            except Exception as e:
                print(f"  Error: {str(e)}")
                continue

        print("\n" + "=" * 60)
        print("DEEP VISION MERGE COMPLETE")
        print("=" * 60)
        print(f"Processed {len(topology_files)} files")
        print(f"Successfully merged {success_count} files")
        print(f"Total chunks updated: {total_updated}")

    def merge_sheet_categories(self):
        """
        Merge per-sheet title block categories into the vector knowledge base.

        Every sheet receives a category from enriched title blocks (when present)
        or deep-vision topology detail names for remaining pages.
        """
        print("=" * 60)
        print("MERGING SHEET CATEGORIES INTO KNOWLEDGE BASE")
        print("=" * 60)
        print("(Title block category on every sheet)")

        self.vector_store.initialize_vectorstore()

        from enhanced_topology_loader import EnhancedTopologyLoader

        topology_files = EnhancedTopologyLoader.list_deep_vision_topology_files()
        if not topology_files:
            print("\nNo deep vision topology files found in data/")
            return

        print(f"\nFound {len(topology_files)} projects")

        total_updated = 0
        success_count = 0

        for topology_file in topology_files:
            file_name = topology_file.stem.replace("enhanced_topology_", "") + ".pdf"
            print(f"\nProcessing: {file_name}")
            try:
                updated = self.vector_store.merge_sheet_categories(file_name)
                total_updated += updated
                if updated > 0:
                    success_count += 1
            except Exception as e:
                print(f"  Error: {str(e)}")
                continue

        print("\n" + "=" * 60)
        print("SHEET CATEGORY MERGE COMPLETE")
        print("=" * 60)
        print(f"Processed {len(topology_files)} files")
        print(f"Successfully merged {success_count} files")
        print(f"Total chunks updated: {total_updated}")

    def train_search_classifications(self):
        """
        Build training artifact for project/page/detail classification taxonomy.

        Loads examples from Training Materials/search classifications.md and writes
        parsed taxonomy + lexical mappings to data/search_classification_training.json.
        """
        print("=" * 60)
        print("TRAINING SEARCH CLASSIFICATIONS")
        print("=" * 60)

        trainer = SearchClassificationTrainer()
        artifact = trainer.build_training_artifact()
        stats = artifact.get('stats', {})

        print("\n✓ Classification training artifact generated")
        print(f"  Training file: {artifact.get('training_file')}")
        print(f"  Project paths: {stats.get('project_paths', 0)}")
        print(f"  Page paths: {stats.get('page_paths', 0)}")
        print(f"  Project labels: {stats.get('project_labels', 0)}")
        print(f"  Page labels: {stats.get('page_labels', 0)}")
        print(f"  Output: {trainer.cache_file}")

    def build_project_profiles(self, merge_to_vector_store: bool = True):
        """
        Build LLM-synthesized project profiles from GP sheets and topology,
        then merge project-overview chunks into the vector knowledge base.
        """
        print("=" * 60)
        print("BUILDING PROJECT PROFILES")
        print("=" * 60)
        print("(GP layout + topology summary + design narrative)")

        builder = ProjectProfileBuilder()
        result = builder.run(self.vector_store, merge_to_vector_store=merge_to_vector_store)

        print("\n" + "=" * 60)
        print("PROJECT PROFILES COMPLETE")
        print("=" * 60)
        print(f"Profiles built: {result.get('profile_count', 0)}")
        print(f"Merged to vector store: {result.get('merged_count', 0)}")
        print(f"Output directory: {result.get('profiles_dir')}")
        print(f"Manifest: {result.get('manifest_path')}")

    def purge_legacy_feedback(self, rebuild_model: bool = True):
        """Remove feedback rows for PDFs that are no longer in the active project library."""
        print("=" * 60)
        print("PURGING LEGACY SEARCH FEEDBACK")
        print("=" * 60)

        result = purge_legacy_feedback(metadata_manager=self.metadata_manager)
        print(f"\n✓ Active indexed projects: {result.get('active_projects', 0)}")
        print(f"✓ Feedback rows kept: {result.get('kept', 0)}")
        print(f"✓ Legacy feedback rows removed: {result.get('removed', 0)}")

        if rebuild_model:
            self.learn_from_search_feedback()

    def learn_from_search_feedback(self, sync_terms: bool = True):
        """Build feedback model from engineer labels and sync learned terminology."""
        print("=" * 60)
        print("LEARNING FROM SEARCH FEEDBACK")
        print("=" * 60)

        result = learn_from_feedback(sync_terms=sync_terms)

        print("\n✓ Feedback model generated")
        print(f"  Model: {result.get('model_path')}")
        print(f"  Source records: {result.get('source_records', 0)}")
        print(f"  Unique queries: {result.get('query_count', 0)}")
        print(f"  Label counts: {result.get('label_counts', {})}")
        print(f"  Related query links: {result.get('related_query_links', 0)}")

        synced = result.get("terminology_synced") or []
        if synced:
            print("\n✓ Terminology synced from feedback acronyms:")
            for item in synced:
                print(f"  - {item}")


def main():
    """Main entry point for the application."""
    import sys
    
    app = LibrarianApp()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py build [--clear]   - Build knowledge base from scratch")
        print("  python main.py update <file>     - Update a specific file (re-process with Vision)")
        print("  python main.py enrich <file>     - Enrich single file with page classification & title blocks")
        print("  python main.py enrich-kb         - Add enrichment to ALL existing documents (NO API calls)")
        print("  python main.py build-graphs      - Build detail connectivity graphs from enriched metadata (NO API calls)")
        print("  python main.py build-enhanced-topology - Build enhanced topology for all enriched projects (NO API calls)")
        print("  python main.py merge-deep-vision  - Merge deep vision topology into vector knowledge base")
        print("  python main.py merge-sheet-categories - Add title block category to every sheet")
        print("  python main.py index-details     - Index detail nodes in vectors for cross-project search")
        print("  python main.py train-classifications - Build project/page/detail classification training artifact")
        print("  python main.py learn-from-feedback - Build feedback model from search-result labels")
        print("  python main.py purge-legacy-feedback - Drop feedback for retired/non-indexed projects")
        print("  python main.py build-project-profiles - Build project overview profiles for search")
        print("  python main.py search <query>    - Search knowledge base")
        print("  python main.py sync-cloud        - Push vectors, metadata, feedback, and profiles to GCS")
        print("  python main.py clean-excluded    - Remove excluded projects from index and metadata")
        print("  python main.py list              - List all projects")
        print("  python main.py stats             - Show statistics")
        print("  python main.py details <file>    - Show project details")
        return
    
    command = sys.argv[1].lower()
    
    if command == 'build':
        clear = '--clear' in sys.argv
        app.build_knowledge_base(clear_existing=clear)
    
    elif command == 'update':
        if len(sys.argv) < 3:
            print("Error: Please provide a file name or path")
            return
        file_input = ' '.join(sys.argv[2:])
        # Check if it's a full path or just filename
        file_path = Path(file_input)
        if not file_path.exists():
            # Try finding it in the projects folder
            file_path = config.PROJECTS_FOLDER / file_input
        if not file_path.exists():
            print(f"Error: File not found: {file_input}")
            return
        app.update_file(file_path)
    
    elif command == 'enrich':
        if len(sys.argv) < 3:
            print("Error: Please provide a file name or path")
            return
        file_input = ' '.join(sys.argv[2:])
        # Check if it's a full path or just filename
        file_path = Path(file_input)
        if not file_path.exists():
            # Try finding it in the projects folder
            file_path = config.PROJECTS_FOLDER / file_input
        if not file_path.exists():
            print(f"Error: File not found: {file_input}")
            return
        app.enrich_metadata(file_path)
    
    elif command == 'enrich-kb':
        app.enrich_knowledge_base()
    
    elif command == 'build-graphs':
        app.build_detail_graphs()

    elif command == 'build-enhanced-topology':
        app.build_enhanced_topology()

    elif command == 'merge-deep-vision':
        app.merge_deep_vision_topology()

    elif command == 'merge-sheet-categories':
        app.merge_sheet_categories()
    
    elif command == 'index-details':
        app.index_detail_nodes()

    elif command == 'train-classifications':
        app.train_search_classifications()

    elif command == 'learn-from-feedback':
        app.learn_from_search_feedback()

    elif command == 'purge-legacy-feedback':
        app.purge_legacy_feedback()

    elif command == 'build-project-profiles':
        app.build_project_profiles()
    
    elif command == 'search':
        if len(sys.argv) < 3:
            print("Error: Please provide a search query")
            return
        query = ' '.join(sys.argv[2:])
        results = app.search(query)
        app.display_search_results(results)
    
    elif command == 'list':
        app.list_all_projects()
    
    elif command == 'stats':
        app.show_statistics()
    
    elif command == 'details':
        if len(sys.argv) < 3:
            print("Error: Please provide a file name")
            return
        file_name = ' '.join(sys.argv[2:])
        app.get_project_details(file_name)

    elif command == 'clean-excluded':
        app.clean_excluded_projects()

    elif command == 'sync-cloud':
        app.sync_cloud()
    
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
