"""Search agent module for intelligent querying of the knowledge base."""
from typing import List, Dict, Optional
from collections import defaultdict
from openai import OpenAI

import config
from vector_store import VectorStore
from metadata_manager import MetadataManager
from engineering_terminology import expand_query


class SearchAgent:
    """Intelligent search agent using RAG and OpenAI."""
    
    def __init__(self):
        """Initialize the search agent."""
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.vector_store = VectorStore()
        self.metadata_manager = MetadataManager()
        
        # Initialize vector store
        self.vector_store.initialize_vectorstore()
    
    def search(
        self,
        query: str,
        k: int = 50,
        category_filter: Optional[str] = None,
        phase_filter: Optional[str] = None,
        use_query_expansion: bool = True
    ) -> Dict:
        """
        Perform semantic search across the knowledge base.
        
        Args:
            query: Search query or keywords
            k: Number of results to retrieve (default: 50)
            category_filter: Optional category filter
            phase_filter: Optional phase filter
            use_query_expansion: Whether to expand query with engineering synonyms (default: True)
            
        Returns:
            Dictionary containing search results and metadata
        """
        print(f"\nSearching for: '{query}'")
        
        # Expand query with engineering terminology synonyms
        expanded_query = query
        if use_query_expansion:
            expanded_terms = expand_query(query, max_expansions=3)
            if len(expanded_terms) > 1:
                # Combine terms for semantic search (embedding will handle similarity)
                expanded_query = " ".join(expanded_terms)
                print(f"Expanded query: '{expanded_query}'")
        
        # Build filter dictionary
        filter_dict = {}
        if category_filter:
            # Note: We'd need to enhance vector_store metadata to include categories
            pass
        if phase_filter:
            filter_dict['phase'] = phase_filter
        
        # Perform vector similarity search with expanded query
        search_results = self.vector_store.similarity_search_with_scores(
            query=expanded_query,
            k=k,
            filter_dict=filter_dict if filter_dict else None
        )
        
        # Organize results by project
        projects_data = self._organize_results_by_project(search_results)
        
        # Enrich with metadata
        enriched_results = self._enrich_with_metadata(projects_data)
        
        # Generate summary using LLM
        summary = self._generate_search_summary(query, enriched_results)
        
        return {
            'query': query,
            'summary': summary,
            'results': enriched_results,
            'total_projects': len(enriched_results),
        }
    
    def _organize_results_by_project(
        self,
        search_results: List[tuple]
    ) -> Dict:
        """
        Organize search results by project file.
        
        Args:
            search_results: List of (document, score) tuples
            
        Returns:
            Dictionary organized by project file name
        """
        projects = defaultdict(lambda: {
            'pages': set(),
            'chunks': [],
            'best_score': float('inf'),
        })
        
        for doc, score in search_results:
            file_name = doc.metadata.get('file_name')
            page = doc.metadata.get('page')
            
            if file_name:
                projects[file_name]['pages'].add(page)
                projects[file_name]['chunks'].append({
                    'content': doc.page_content,
                    'page': page,
                    'score': score,
                })
                
                # Track best (lowest) score
                if score < projects[file_name]['best_score']:
                    projects[file_name]['best_score'] = score
        
        # Convert sets to sorted lists
        for file_name in projects:
            projects[file_name]['pages'] = sorted(list(projects[file_name]['pages']))
        
        return dict(projects)
    
    def _enrich_with_metadata(self, projects_data: Dict) -> List[Dict]:
        """
        Enrich project data with metadata.
        
        Args:
            projects_data: Organized search results by project
            
        Returns:
            List of enriched project results
        """
        enriched = []
        
        for file_name, data in projects_data.items():
            # Get metadata from metadata manager
            metadata = self.metadata_manager.get_project_metadata(file_name)
            
            if metadata:
                enriched.append({
                    'project_name': metadata.get('project_name', file_name),
                    'file_name': file_name,
                    'file_path': metadata.get('file_path', ''),
                    'phase': metadata.get('phase', 'Unknown'),
                    'engineer_of_record': metadata.get('engineer_of_record', 'Unknown'),
                    'date': metadata.get('date', 'Unknown'),
                    'categories': metadata.get('categories', []),
                    'relevant_pages': data['pages'],
                    'total_pages': metadata.get('total_pages', 0),
                    'relevance_score': data['best_score'],
                    'sample_content': data['chunks'][0]['content'][:300] + '...' if data['chunks'] else '',
                    'chunks': data['chunks'],  # Include all chunks with their page-specific scores
                })
            else:
                # Fallback if metadata not found
                enriched.append({
                    'project_name': file_name,
                    'file_name': file_name,
                    'file_path': '',
                    'phase': 'Unknown',
                    'engineer_of_record': 'Unknown',
                    'date': 'Unknown',
                    'categories': [],
                    'relevant_pages': data['pages'],
                    'total_pages': 0,
                    'relevance_score': data['best_score'],
                    'sample_content': data['chunks'][0]['content'][:300] + '...' if data['chunks'] else '',
                    'chunks': data['chunks'],  # Include all chunks with their page-specific scores
                })
        
        # Sort by relevance score (lower is better)
        enriched.sort(key=lambda x: x['relevance_score'])
        
        return enriched
    
    def _generate_search_summary(
        self,
        query: str,
        results: List[Dict]
    ) -> str:
        """
        Generate a natural language summary of search results using LLM.
        
        Args:
            query: Original search query
            results: Enriched search results
            
        Returns:
            Summary string
        """
        if not results:
            return "No relevant documents found for the query."
        
        # Prepare context for LLM
        context = f"Search Query: {query}\n\n"
        context += f"Found {len(results)} relevant project(s):\n\n"
        
        for i, result in enumerate(results[:5], 1):  # Top 5 results
            context += f"{i}. Project: {result['project_name']}\n"
            context += f"   Phase: {result['phase']}\n"
            context += f"   Engineer: {result['engineer_of_record']}\n"
            context += f"   Relevant Pages: {', '.join(map(str, result['relevant_pages'][:10]))}\n"
            context += f"   Sample: {result['sample_content'][:200]}\n\n"
        
        # Generate summary
        try:
            response = self.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that summarizes search results for structural engineering projects. Provide a concise summary highlighting the most relevant findings."
                    },
                    {
                        "role": "user",
                        "content": f"Summarize these search results:\n\n{context}"
                    }
                ],
                temperature=0.5,
                max_tokens=300
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            return f"Found {len(results)} relevant project(s). See results below."
    
    def advanced_search(
        self,
        query: str,
        filters: Optional[Dict] = None
    ) -> Dict:
        """
        Perform advanced search with multiple filters.
        
        Args:
            query: Search query
            filters: Dictionary of filters (category, phase, engineer, date_range)
            
        Returns:
            Search results
        """
        filters = filters or {}
        
        # Start with semantic search
        k = filters.get('max_results', 30)
        results = self.search(
            query=query,
            k=k,
            category_filter=filters.get('category'),
            phase_filter=filters.get('phase')
        )
        
        # Apply additional filters
        filtered_results = results['results']
        
        if filters.get('engineer'):
            engineer = filters['engineer'].lower()
            filtered_results = [
                r for r in filtered_results
                if engineer in r['engineer_of_record'].lower()
            ]
        
        if filters.get('category'):
            category = filters['category']
            filtered_results = [
                r for r in filtered_results
                if category in r['categories']
            ]
        
        # Update results
        results['results'] = filtered_results
        results['total_projects'] = len(filtered_results)
        
        return results
    
    def get_project_details(self, file_name: str) -> Optional[Dict]:
        """Get detailed information about a specific project."""
        return self.metadata_manager.get_project_metadata(file_name)
    
    def list_all_projects(self) -> List[Dict]:
        """List all indexed projects."""
        projects = self.metadata_manager.get_all_projects()
        return list(projects.values())
    
    def get_statistics(self) -> Dict:
        """Get statistics about the knowledge base."""
        metadata_stats = self.metadata_manager.get_stats()
        vector_info = self.vector_store.get_collection_info()
        
        return {
            'metadata': metadata_stats,
            'vector_store': vector_info,
        }
