#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Custom Pipeline Visualization Module for TOF_ML

This module creates beautiful, custom SVG visualizations of pipeline provenance data,
optimized specifically for machine learning pipelines with a focus on
data transformations, splits, and model training.
"""

import os
import json
import logging
import datetime
import html
from typing import Dict, Any, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class PipelineVisualizer:
    """
    Creates custom SVG visualizations of ML pipeline provenance data.
    
    Features:
    - Beautiful, clean design with custom styling
    - Optimized layout for ML pipeline visualization
    - Detailed metadata for each node
    - Performance metrics and timing information
    """
    
    def __init__(self, provenance_path: str, output_dir: str):
        """
        Initialize the pipeline visualizer.
        
        Args:
            provenance_path: Path to the provenance data (lineage_graph.json)
            output_dir: Directory to save visualizations
        """
        self.provenance_path = provenance_path
        self.output_dir = output_dir
        
        # Load provenance data
        with open(provenance_path, 'r') as f:
            self.provenance_data = json.load(f)
        
        # Extract components
        self.lineage_graph = self.provenance_data.get("lineage_graph", {})
        self.data_sources = self.provenance_data.get("data_sources", {})
        self.transformations = self.provenance_data.get("transformations", {})
        self.data_splits = self.provenance_data.get("data_splits", {})
        self.operations = self.provenance_data.get("operations", {})
        self.run_id = self.provenance_data.get("run_id", "Unknown")
        self.start_time = self.provenance_data.get("start_time", "Unknown")
        self.end_time = self.provenance_data.get("end_time", "Unknown")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Color scheme
        self.colors = {
            "background": "#f8f9fa",
            "data_source": {"fill": "#b3e0ff", "stroke": "#2980b9"},
            "transformation": {"fill": "#abebc6", "stroke": "#27ae60"},
            "transform_item": {"fill": "#d5f5e3", "stroke": "#27ae60"},
            "data_split": {"fill": "#d6eaf8", "stroke": "#3498db"},
            "train_data": {"fill": "#fad7a0", "stroke": "#e67e22"},
            "validation_data": {"fill": "#d7bde2", "stroke": "#8e44ad"},
            "test_data": {"fill": "#f5b7b1", "stroke": "#c0392b"},
            "model": {"fill": "#ebdef0", "stroke": "#8e44ad"},
            "text": "#333333",
            "text_light": "#777777"
        }
        
        logger.info(f"PipelineVisualizer initialized with {len(self.lineage_graph)} lineage nodes")
    
    def create_pipeline_visualization(self, output_path: Optional[str] = None) -> str:
        """
        Create a beautiful visualization of the ML pipeline.
        
        Args:
            output_path: Optional path to save the visualization
            
        Returns:
            Path to the saved visualization file
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, "pipeline_visualization.svg")
        
        # Analyze pipeline structure
        sources = self._find_data_sources()
        transformations = self._find_transformations()
        splits = self._find_data_splits()
        train_datasets = self._find_nodes_by_type("train_data")
        val_datasets = self._find_nodes_by_type("validation_data")
        test_datasets = self._find_nodes_by_type("test_data")
        models = self._find_nodes_by_type("trained_model")
        
        # SVG dimensions
        width = 900
        height = 600
        
        # Create SVG content
        svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">
  <!-- Background -->
  <rect width="{width}" height="{height}" fill="{self.colors['background']}" />
  
  <!-- Title -->
  <text x="{width/2}" y="30" font-family="Arial" font-size="20" font-weight="bold" text-anchor="middle">ML Pipeline Visualization</text>
  <text x="{width/2}" y="55" font-family="Arial" font-size="12" text-anchor="middle">Run ID: {self.run_id}</text>
  
  <!-- Arrowhead markers -->
  <defs>
    {self._create_arrow_markers()}
  </defs>
  
  {self._create_pipeline_elements(sources, transformations, splits, train_datasets, val_datasets, test_datasets, models, width, height)}
  
  {self._create_timeline(width, height)}
  
  {self._create_legend(width, height)}
</svg>"""
        
        # Write to file with explicit UTF-8 encoding
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        
        logger.info(f"Pipeline visualization saved to {output_path}")
        return output_path
    
    def _create_arrow_markers(self) -> str:
        """Create SVG marker definitions for arrows."""
        markers = """
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333" />
    </marker>
    <marker id="arrowhead-transform" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#27ae60" />
    </marker>
    <marker id="arrowhead-train" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#e67e22" />
    </marker>
    <marker id="arrowhead-val" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#8e44ad" />
    </marker>
    <marker id="arrowhead-test" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#c0392b" />
    </marker>
    <marker id="arrowhead-model" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#8e44ad" />
    </marker>"""
        return markers
    
    def _find_data_sources(self) -> List[str]:
        """Find all data source nodes in the lineage graph."""
        sources = [node_id for node_id, info in self.lineage_graph.items() 
                if info.get("type") == "data_source"]
        
        # If no data sources found, check if we can infer one from the data_sources dictionary
        if not sources and self.data_sources:
            # Use the first data source ID as a fallback
            sources = [next(iter(self.data_sources.keys()))]
            # Add this to the lineage graph to ensure it's properly handled
            self.lineage_graph[sources[0]] = {"type": "data_source"}
            logger.info(f"No explicit data source found in lineage graph, using inferred source: {sources[0]}")
        
        # If still no sources, create a default placeholder
        if not sources:
            default_id = "default_data_source"
            sources = [default_id]
            # Add to lineage graph and data_sources for proper handling
            self.lineage_graph[default_id] = {"type": "data_source"}
            self.data_sources[default_id] = {
                "source_info": {
                    "loader_type": "Unknown",
                    "raw_data_shape": {"shape": [0, 0]}
                }
            }
            logger.info("No data sources found, using default placeholder")
        
        return sources
    
    def _find_transformations(self) -> List[Dict[str, Any]]:
        """Find all transformation nodes in the lineage graph."""
        transform_items = []
        for transform_id, transform_data in self.transformations.items():
            input_id = transform_data.get("input_data_id")
            output_id = transform_data.get("output_data_id")
            transform_info = transform_data.get("transformation_info", {})
            transform_items.append({
                "id": transform_id,
                "input_id": input_id,
                "output_id": output_id,
                "type": transform_info.get("transformer_type", "Unknown"),
                "timestamp": transform_data.get("timestamp")
            })
        
        # Sort by timestamp
        transform_items.sort(key=lambda x: x.get("timestamp", ""))
        return transform_items
    
    def _find_data_splits(self) -> List[Dict[str, Any]]:
        """Find all data split operations."""
        split_items = []
        for split_id, split_data in self.data_splits.items():
            split_items.append({
                "id": split_id,
                "input_id": split_data.get("input_data_id"),
                "train_id": split_data.get("train_data_id"),
                "val_id": split_data.get("val_data_id"),
                "test_id": split_data.get("test_data_id"),
                "split_info": split_data.get("split_info", {}),
                "timestamp": split_data.get("timestamp")
            })
        return split_items
    
    def _find_nodes_by_type(self, node_type: str) -> List[str]:
        """Find all nodes of a specific type in the lineage graph."""
        return [node_id for node_id, info in self.lineage_graph.items() 
                if info.get("type") == node_type]
    
    def _create_pipeline_elements(self, 
                                 sources: List[str], 
                                 transformations: List[Dict[str, Any]], 
                                 splits: List[Dict[str, Any]],
                                 train_datasets: List[str],
                                 val_datasets: List[str],
                                 test_datasets: List[str],
                                 models: List[str],
                                 width: int,
                                 height: int) -> str:
        """Create SVG elements for pipeline visualization."""
        elements = []
        
        # Layout calculations
        left_margin = 50
        right_margin = 50
        usable_width = width - left_margin - right_margin
        
        # Source node dimensions
        source_width = 200
        source_height = 100
        source_x = left_margin
        source_y = 120
        
        # Create source nodes
        if sources:
            for i, source_id in enumerate(sources):
                source_info = self.data_sources.get(source_id, {})
                source_metadata = source_info.get("source_info", {})
                
                # Extract shape information
                shape_info = source_metadata.get("raw_data_shape", {})
                if "shape" in shape_info:
                    shape_text = f"{shape_info['shape'][0]} × {shape_info['shape'][1]}"
                else:
                    shape_text = "Unknown shape"
                
                elements.append(self._create_node(
                    x=source_x,
                    y=source_y + i * (source_height + 20),
                    width=source_width,
                    height=source_height,
                    node_type="data_source",
                    title="Raw Data",
                    content=[
                        f"Loader: {source_metadata.get('loader_type', 'Unknown')}",
                        f"Shape: {shape_text}",
                        self._truncate_id(source_id)
                    ]
                ))
        else:
            # Create a default data source node if none exists
            # Try to get data shape from data_manager metadata 
            shape_text = "Unknown"
            for op_id, op_data in self.operations.items():
                # Check if this is a loading operation
                if isinstance(op_data.get("operation_name", ""), str) and "load" in op_data.get("operation_name", "").lower():
                    # Look for metadata
                    metadata = op_data.get("metadata", {})
                    if metadata and "shape" in metadata:
                        shape = metadata["shape"]
                        if isinstance(shape, list) and len(shape) >= 2:
                            shape_text = f"{shape[0]} × {shape[1]}"
                            break
            
            elements.append(self._create_node(
                x=source_x,
                y=source_y,
                width=source_width,
                height=source_height,
                node_type="data_source",
                title="Raw Data",
                content=[
                    "Loader: EnhancedH5Loader",
                    f"Shape: {shape_text}",
                    "raw_data"
                ]
            ))
            logger.info("Created default data source node with shape: " + shape_text)
        
        # Transformation group
        transform_x = source_x + source_width + 100
        transform_y = source_y
        transform_width = 200
        transform_height = 140
        
        # Add transformation group
        elements.append(f"""
  <g>
    <!-- Transformation Group -->
    <rect x="{transform_x}" y="{transform_y}" width="{transform_width}" height="{transform_height}" rx="5" ry="5" 
         fill="{self.colors['transformation']['fill']}" stroke="{self.colors['transformation']['stroke']}" stroke-width="2" />
    <text x="{transform_x + transform_width/2}" y="{transform_y + 25}" font-family="Arial" font-size="14" 
         font-weight="bold" text-anchor="middle">Transformations</text>
  </g>""")
        
        # Add individual transformations
        for i, transform in enumerate(transformations):
            transform_item_height = 40
            transform_item_width = transform_width - 40
            transform_item_x = transform_x + 20
            transform_item_y = transform_y + 40 + i * (transform_item_height + 10)
            
            transform_info = self.transformations.get(transform["id"], {}).get("transformation_info", {})
            config = transform_info.get("transformer_config", {})
            
            # Get columns if available and sanitize
            columns = config.get("columns", [])
            if columns:
                # Convert list to string and sanitize
                columns_str = str(columns).replace("'", "").replace("[", "").replace("]", "")
                columns_text = self._sanitize_text(f"columns: {columns_str}")
            else:
                columns_text = ""
            
            # Sanitize transform type
            safe_transform_type = self._sanitize_text(transform["type"])
            
            elements.append(f"""
  <g>
    <!-- Transformation: {safe_transform_type} -->
    <rect x="{transform_item_x}" y="{transform_item_y}" width="{transform_item_width}" height="{transform_item_height}" rx="3" ry="3" 
         fill="{self.colors['transform_item']['fill']}" stroke="{self.colors['transform_item']['stroke']}" stroke-width="1" />
    <text x="{transform_item_x + transform_item_width/2}" y="{transform_item_y + 20}" font-family="Arial" font-size="12" 
         text-anchor="middle">{safe_transform_type}</text>
    <text x="{transform_item_x + transform_item_width/2}" y="{transform_item_y + 35}" font-family="Arial" font-size="10" 
         text-anchor="middle">{columns_text}</text>
  </g>""")
        
        # Data split
        split_x = transform_x
        split_y = transform_y + transform_height + 30
        split_width = 200
        split_height = 100
        
        for i, split in enumerate(splits):
            split_info = split.get("split_info", {})
            splitter_type = split_info.get("splitter_type", "Unknown")
            
            # Get split ratios
            train_ratio = split_info.get("train_ratio", 0)
            val_ratio = split_info.get("val_ratio", 0)
            test_ratio = split_info.get("test_ratio", 0)
            
            elements.append(self._create_node(
                x=split_x,
                y=split_y + i * (split_height + 20),
                width=split_width,
                height=split_height,
                node_type="data_split",
                title="Data Split",
                content=[
                    f"{splitter_type}",
                    f"train: {train_ratio:.0%}, val: {val_ratio:.0%}, test: {test_ratio:.0%}",
                    self._truncate_id(split["id"])
                ]
            ))
        
        # Output datasets
        output_x = transform_x + transform_width + 100
        output_width = 150
        output_height = 80
        
        # Add train datasets
        if train_datasets:
            train_y = source_y
            train_id = train_datasets[0]
            
            # Get shape info from split info
            split_info = splits[0].get("split_info", {}) if splits else {}
            train_shape = split_info.get("train_shape", {})
            
            if "features_shape" in train_shape:
                shape_text = f"{train_shape['features_shape'][0]} × {train_shape['features_shape'][1]} features"
            else:
                shape_text = "Unknown shape"
            
            elements.append(self._create_node(
                x=output_x,
                y=train_y,
                width=output_width,
                height=output_height,
                node_type="train_data",
                title="Train Data",
                content=[
                    shape_text,
                    self._truncate_id(train_id)
                ]
            ))
        
        # Add validation datasets
        if val_datasets:
            val_y = source_y + 110
            val_id = val_datasets[0]
            
            # Get shape info from split info
            split_info = splits[0].get("split_info", {}) if splits else {}
            val_shape = split_info.get("val_shape", {})
            
            if "features_shape" in val_shape:
                shape_text = f"{val_shape['features_shape'][0]} × {val_shape['features_shape'][1]} features"
            else:
                shape_text = "Unknown shape"
            
            elements.append(self._create_node(
                x=output_x,
                y=val_y,
                width=output_width,
                height=output_height,
                node_type="validation_data",
                title="Validation Data",
                content=[
                    shape_text,
                    self._truncate_id(val_id)
                ]
            ))
        
        # Add test datasets
        if test_datasets:
            test_y = source_y + 220
            test_id = test_datasets[0]
            
            # Get shape info from split info
            split_info = splits[0].get("split_info", {}) if splits else {}
            test_shape = split_info.get("test_shape", {})
            
            if "features_shape" in test_shape:
                shape_text = f"{test_shape['features_shape'][0]} × {test_shape['features_shape'][1]} features"
            else:
                shape_text = "Unknown shape"
            
            elements.append(self._create_node(
                x=output_x,
                y=test_y,
                width=output_width,
                height=output_height,
                node_type="test_data",
                title="Test Data",
                content=[
                    shape_text,
                    self._truncate_id(test_id)
                ]
            ))
        
        # Add connections
        elements.extend(self._create_connections(
            sources, transformations, splits, 
            train_datasets, val_datasets, test_datasets,
            source_x, source_y, source_width, source_height,
            transform_x, transform_y, transform_width, transform_height,
            split_x, split_y, split_width, split_height,
            output_x, output_width, output_height
        ))
        
        return '\n'.join(elements)
    
    def _create_node(self, x: int, y: int, width: int, height: int, 
                    node_type: str, title: str, content: List[str]) -> str:
        """Create an SVG node with title and content."""
        colors = self.colors[node_type]
        
        # Sanitize title and content
        safe_title = self._sanitize_text(title)
        
        node = f"""
  <g>
    <rect x="{x}" y="{y}" width="{width}" height="{height}" rx="5" ry="5" 
         fill="{colors['fill']}" stroke="{colors['stroke']}" stroke-width="2" />
    <text x="{x + width/2}" y="{y + 25}" font-family="Arial" font-size="14" 
         font-weight="bold" text-anchor="middle">{safe_title}</text>"""
        
        # Add content lines
        for i, line in enumerate(content):
            # Sanitize each line
            safe_line = self._sanitize_text(line)
            node += f"""
    <text x="{x + width/2}" y="{y + 45 + i*20}" font-family="Arial" font-size="{12 if i < len(content)-1 else 10}" 
         text-anchor="middle" {f'fill="{self.colors["text_light"]}"' if i == len(content)-1 else ''}>{safe_line}</text>"""
        
        node += """
  </g>"""
        
        return node
    
    def _truncate_id(self, id_str: str, length: int = 16) -> str:
        """Truncate long ID strings for display."""
        if len(id_str) > length:
            half = length // 2
            return f"{id_str[:half]}...{id_str[-half:]}"
        return id_str
        
    def _sanitize_text(self, text: str) -> str:
        """Sanitize text for XML/SVG content to prevent encoding issues."""
        if text is None:
            return ""
        # First escape HTML/XML entities
        sanitized = html.escape(str(text))
        # Replace non-printable characters
        sanitized = ''.join(c if c.isprintable() or c.isspace() else '?' for c in sanitized)
        return sanitized
    
    def _create_connections(self,
                          sources: List[str],
                          transformations: List[Dict[str, Any]],
                          splits: List[Dict[str, Any]],
                          train_datasets: List[str],
                          val_datasets: List[str],
                          test_datasets: List[str],
                          source_x: int, source_y: int, source_width: int, source_height: int,
                          transform_x: int, transform_y: int, transform_width: int, transform_height: int,
                          split_x: int, split_y: int, split_width: int, split_height: int,
                          output_x: int, output_width: int, output_height: int) -> List[str]:
        """Create connection lines between nodes."""
        connections = []
        
        # Source to Transformations - Always add this connection
        connections.append(f"""
  <!-- Source to Transformations -->
  <path d="M{source_x + source_width},{source_y + source_height/2} L{transform_x},{transform_y + transform_height/2}" 
       stroke="#333" stroke-width="2" fill="none" marker-end="url(#arrowhead)"/>""")
        
        # Log that we're adding this connection
        logger.debug(f"Added connection from source ({source_x + source_width},{source_y + source_height/2}) to transform ({transform_x},{transform_y + transform_height/2})")
        
        # Transformations to Split
        connections.append(f"""
  <!-- Transformations to Split -->
  <path d="M{transform_x + transform_width/2},{transform_y + transform_height} L{transform_x + transform_width/2},{split_y}" 
       stroke="#333" stroke-width="2" fill="none" marker-end="url(#arrowhead)"/>""")
        
        # Calculate output positions
        train_y = source_y + output_height/2
        val_y = source_y + 110 + output_height/2
        test_y = source_y + 220 + output_height/2
        
        # Split to Train Data
        if train_datasets:
            connections.append(f"""
  <!-- Split to Train Data -->
  <path d="M{split_x + split_width},{split_y + split_height/3} L{split_x + split_width + 30},{split_y + split_height/3} L{split_x + split_width + 30},{train_y} L{output_x},{train_y}" 
       stroke="{self.colors['train_data']['stroke']}" stroke-width="2" fill="none" marker-end="url(#arrowhead-train)"/>
  <text x="{split_x + split_width + 40}" y="{train_y - 15}" font-family="Arial" font-size="10" 
       fill="{self.colors['train_data']['stroke']}" text-anchor="middle">train split</text>""")
        
        # Split to Validation Data
        if val_datasets:
            connections.append(f"""
  <!-- Split to Validation Data -->
  <path d="M{split_x + split_width},{split_y + split_height/2} L{split_x + split_width + 30},{split_y + split_height/2} L{split_x + split_width + 30},{val_y} L{output_x},{val_y}" 
       stroke="{self.colors['validation_data']['stroke']}" stroke-width="2" fill="none" marker-end="url(#arrowhead-val)"/>
  <text x="{split_x + split_width + 40}" y="{val_y - 15}" font-family="Arial" font-size="10" 
       fill="{self.colors['validation_data']['stroke']}" text-anchor="middle">validation split</text>""")
        
        # Split to Test Data
        if test_datasets:
            connections.append(f"""
  <!-- Split to Test Data -->
  <path d="M{split_x + split_width},{split_y + 2*split_height/3} L{split_x + split_width + 30},{split_y + 2*split_height/3} L{split_x + split_width + 30},{test_y} L{output_x},{test_y}" 
       stroke="{self.colors['test_data']['stroke']}" stroke-width="2" fill="none" marker-end="url(#arrowhead-test)"/>
  <text x="{split_x + split_width + 40}" y="{test_y - 15}" font-family="Arial" font-size="10" 
       fill="{self.colors['test_data']['stroke']}" text-anchor="middle">test split</text>""")
        
        return connections
    
    def _create_timeline(self, width: int, height: int) -> str:
        """Create a timeline of operations."""
        # Calculate durations
        durations = {}
        total_duration = 0
        
        for op_id, op_data in self.operations.items():
            # Convert keys to lowercase to handle case inconsistencies
            op_data_lower = {k.lower(): v for k, v in op_data.items()}
            
            op_name = op_data_lower.get("operation_name", "")
            if isinstance(op_name, str) and op_name.lower().startswith("transform"):
                op_name = "transformation"
            
            # Try various keys for duration in case of inconsistency
            duration = op_data_lower.get("duration_seconds", 0)
            if duration == 0:
                duration = op_data_lower.get("duration", 0)
            
            # Normalize operation names
            if "load" in op_name.lower():
                op_name = "data_loading"
            elif "transform" in op_name.lower():
                op_name = "transformation"
            elif "split" in op_name.lower():
                op_name = "data_splitting"
            
            if op_name in durations:
                durations[op_name] += duration
            else:
                durations[op_name] = duration
            
            total_duration += duration
        
        # Format duration strings
        duration_strings = []
        for op_name, duration in durations.items():
            # Clean up operation name and sanitize
            clean_name = op_name.replace("_", " ").title()
            safe_name = self._sanitize_text(clean_name)
            duration_strings.append(f"{safe_name}: {duration:.2f}s")
        
        timeline = f"""
  <!-- Duration -->
  <g>
    <text x="{width/2}" y="{height - 70}" font-family="Arial" font-size="12" text-anchor="middle">{' | '.join(duration_strings)}</text>
    <text x="{width/2}" y="{height - 50}" font-family="Arial" font-size="12" text-anchor="middle">Total Pipeline Duration: {total_duration:.2f}s</text>
  </g>"""
        
        return timeline
    
    def _create_legend(self, width: int, height: int) -> str:
        """Create a legend for the visualization."""
        legend = f"""
  <!-- Legend -->
  <g>
    <rect x="50" y="{height - 80}" width="120" height="60" rx="3" ry="3" fill="{self.colors['background']}" stroke="#ddd" stroke-width="1" />
    <text x="110" y="{height - 65}" font-family="Arial" font-size="10" font-weight="bold" text-anchor="middle">Legend</text>
    
    <rect x="60" y="{height - 55}" width="12" height="12" fill="{self.colors['data_source']['fill']}" stroke="{self.colors['data_source']['stroke']}" stroke-width="1" />
    <text x="77" y="{height - 45}" font-family="Arial" font-size="9" text-anchor="start">Data Source</text>
    
    <rect x="60" y="{height - 40}" width="12" height="12" fill="{self.colors['transformation']['fill']}" stroke="{self.colors['transformation']['stroke']}" stroke-width="1" />
    <text x="77" y="{height - 30}" font-family="Arial" font-size="9" text-anchor="start">Transformation</text>
    
    <rect x="60" y="{height - 25}" width="12" height="12" fill="{self.colors['data_split']['fill']}" stroke="{self.colors['data_split']['stroke']}" stroke-width="1" />
    <text x="77" y="{height - 15}" font-family="Arial" font-size="9" text-anchor="start">Data Split</text>
  </g>"""
        
        return legend

def visualize_provenance_graph(provenance_path: str, output_path: str) -> str:
    """
    Create a beautiful visualization of the provenance graph.
    This function is a direct replacement for ProvenanceTracker.visualize_provenance_graph.
    
    Args:
        provenance_path: Path to the provenance graph JSON file
        output_path: Path to save the visualization
        
    Returns:
        Path to the saved visualization
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualizer
    visualizer = PipelineVisualizer(provenance_path, output_dir)
    
    # Create visualization (using basename without extension as the filename)
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    return visualizer.create_pipeline_visualization(output_path)
