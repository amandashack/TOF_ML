#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Provenance Tracker for the ML Provenance Tracker Framework.
This module tracks data lineage and transformations.
"""

import os
import logging
import json
import time
import datetime
import uuid
import contextlib
from typing import Dict, Any, Optional, List, Tuple, Union, ContextManager

logger = logging.getLogger(__name__)

class ProvenanceTracker:
    """
    Tracks data provenance throughout the ML pipeline.
    
    This class records:
    - Data sources and their metadata
    - Transformations applied to data
    - Data splits
    - Model training inputs
    - Full lineage (data -> transformations -> models -> outputs)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the provenance tracker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.provenance_enabled = config.get("provenance", {}).get("enabled", True)
        self.provenance_db_path = config.get("provenance", {}).get("db_path", "./provenance")
        
        # Create provenance storage
        self.data_sources = {}
        self.transformations = {}
        self.data_splits = {}
        self.operations = {}
        self.lineage_graph = {}
        
        # Initialize provenance DB
        if self.provenance_enabled:
            os.makedirs(self.provenance_db_path, exist_ok=True)
        
        # Generate a unique run ID
        self.run_id = str(uuid.uuid4())
        self.start_time = datetime.datetime.now().isoformat()
        
        logger.info(f"ProvenanceTracker initialized with run_id: {self.run_id}")
    
    @contextlib.contextmanager
    def track_operation(self, operation_name: str) -> ContextManager:
        """
        Context manager to track an operation's execution time and metadata.
        
        Args:
            operation_name: Name of the operation to track
            
        Yields:
            Context for the operation
        """
        if not self.provenance_enabled:
            yield
            return
        
        # Generate operation ID
        operation_id = f"{operation_name}_{uuid.uuid4()}"
        
        # Record start
        start_time = time.time()
        logger.debug(f"Starting operation: {operation_name} (id: {operation_id})")
        
        try:
            # Yield control back to the caller
            yield
            
            # Record success
            status = "completed"
            error = None
        except Exception as e:
            # Record failure
            status = "failed"
            error = str(e)
            raise
        finally:
            # Record end time and duration
            end_time = time.time()
            duration = end_time - start_time
            
            # Store operation metadata
            self.operations[operation_id] = {
                "operation_name": operation_name,
                "start_time": datetime.datetime.fromtimestamp(start_time).isoformat(),
                "end_time": datetime.datetime.fromtimestamp(end_time).isoformat(),
                "duration_seconds": duration,
                "status": status,
                "error": error,
                "run_id": self.run_id
            }
            
            logger.debug(f"Finished operation: {operation_name} in {duration:.2f}s (status: {status})")
    
    def record_data_source(self, data_id: str, source_info: Dict[str, Any]) -> None:
        """
        Record information about a data source.
        
        Args:
            data_id: Unique identifier for the data
            source_info: Information about the data source
        """
        if not self.provenance_enabled:
            return
        
        logger.debug(f"Recording data source: {data_id}")
        
        # Store data source information
        self.data_sources[data_id] = {
            "data_id": data_id,
            "source_info": source_info,
            "timestamp": datetime.datetime.now().isoformat(),
            "run_id": self.run_id
        }
        
        # Add to lineage graph
        self.lineage_graph[data_id] = {
            "type": "data_source",
            "inputs": [],
            "outputs": []
        }
        
        # If we're saving to disk, write the provenance record
        if self.provenance_db_path:
            self._save_provenance_record("data_sources", data_id, self.data_sources[data_id])
    
    def record_transformation(
        self, 
        input_data_id: str, 
        output_data_id: str, 
        transformation_info: Dict[str, Any]
    ) -> None:
        """
        Record information about a data transformation.
        
        Args:
            input_data_id: ID of the input data
            output_data_id: ID of the output data
            transformation_info: Information about the transformation
        """
        if not self.provenance_enabled:
            return
        
        logger.debug(f"Recording transformation: {input_data_id} -> {output_data_id}")
        
        # Generate transformation ID
        transformation_id = f"transform_{uuid.uuid4()}"
        
        # Store transformation information
        self.transformations[transformation_id] = {
            "transformation_id": transformation_id,
            "input_data_id": input_data_id,
            "output_data_id": output_data_id,
            "transformation_info": transformation_info,
            "timestamp": datetime.datetime.now().isoformat(),
            "run_id": self.run_id
        }
        
        # Update lineage graph
        if input_data_id in self.lineage_graph:
            self.lineage_graph[input_data_id]["outputs"].append({
                "id": output_data_id,
                "via_transformation": transformation_id
            })
        
        self.lineage_graph[output_data_id] = {
            "type": "transformed_data",
            "inputs": [{
                "id": input_data_id,
                "via_transformation": transformation_id
            }],
            "outputs": []
        }
        
        # Save to disk
        if self.provenance_db_path:
            self._save_provenance_record("transformations", transformation_id, self.transformations[transformation_id])
    
    def record_data_split(
        self,
        input_data_id: str,
        train_data_id: str,
        val_data_id: str,
        test_data_id: str,
        split_info: Dict[str, Any]
    ) -> None:
        """
        Record information about a data split operation.
        
        Args:
            input_data_id: ID of the input data
            train_data_id: ID of the training data
            val_data_id: ID of the validation data
            test_data_id: ID of the test data
            split_info: Information about the split operation
        """
        if not self.provenance_enabled:
            return
        
        logger.debug(f"Recording data split: {input_data_id} -> {train_data_id}, {val_data_id}, {test_data_id}")
        
        # Generate split ID
        split_id = f"split_{uuid.uuid4()}"
        
        # Store split information
        self.data_splits[split_id] = {
            "split_id": split_id,
            "input_data_id": input_data_id,
            "train_data_id": train_data_id,
            "val_data_id": val_data_id,
            "test_data_id": test_data_id,
            "split_info": split_info,
            "timestamp": datetime.datetime.now().isoformat(),
            "run_id": self.run_id
        }
        
        # Update lineage graph
        if input_data_id in self.lineage_graph:
            self.lineage_graph[input_data_id]["outputs"].extend([
                {"id": train_data_id, "via_split": split_id, "split_type": "train"},
                {"id": val_data_id, "via_split": split_id, "split_type": "validation"},
                {"id": test_data_id, "via_split": split_id, "split_type": "test"}
            ])
        
        # Add split outputs to lineage graph
        for data_id, split_type in [
            (train_data_id, "train"), 
            (val_data_id, "validation"), 
            (test_data_id, "test")
        ]:
            self.lineage_graph[data_id] = {
                "type": f"{split_type}_data",
                "inputs": [{
                    "id": input_data_id,
                    "via_split": split_id,
                    "split_type": split_type
                }],
                "outputs": []
            }
        
        # Save to disk
        if self.provenance_db_path:
            self._save_provenance_record("data_splits", split_id, self.data_splits[split_id])
    
    def record_model_training(
        self,
        model_id: str,
        train_data_id: str,
        val_data_id: str,
        model_info: Dict[str, Any],
        training_info: Dict[str, Any]
    ) -> None:
        """
        Record information about model training.
        
        Args:
            model_id: ID of the trained model
            train_data_id: ID of the training data
            val_data_id: ID of the validation data
            model_info: Information about the model
            training_info: Information about the training process
        """
        if not self.provenance_enabled:
            return
        
        logger.debug(f"Recording model training: {train_data_id}, {val_data_id} -> {model_id}")
        
        # Generate training ID
        training_id = f"training_{uuid.uuid4()}"
        
        # Store training information
        self.operations[training_id] = {
            "operation_id": training_id,
            "operation_name": "model_training",
            "model_id": model_id,
            "train_data_id": train_data_id,
            "val_data_id": val_data_id,
            "model_info": model_info,
            "training_info": training_info,
            "timestamp": datetime.datetime.now().isoformat(),
            "run_id": self.run_id
        }
        
        # Update lineage graph
        if train_data_id in self.lineage_graph:
            self.lineage_graph[train_data_id]["outputs"].append({
                "id": model_id,
                "via_training": training_id,
                "role": "training_data"
            })
        
        if val_data_id in self.lineage_graph:
            self.lineage_graph[val_data_id]["outputs"].append({
                "id": model_id,
                "via_training": training_id,
                "role": "validation_data"
            })
        
        self.lineage_graph[model_id] = {
            "type": "trained_model",
            "inputs": [
                {"id": train_data_id, "via_training": training_id, "role": "training_data"},
                {"id": val_data_id, "via_training": training_id, "role": "validation_data"}
            ],
            "outputs": []
        }
        
        # Save to disk
        if self.provenance_db_path:
            self._save_provenance_record("operations", training_id, self.operations[training_id])
    
    def record_model_evaluation(
        self,
        model_id: str,
        test_data_id: str,
        evaluation_results: Dict[str, Any]
    ) -> None:
        """
        Record information about model evaluation.
        
        Args:
            model_id: ID of the model
            test_data_id: ID of the test data
            evaluation_results: Results of the evaluation
        """
        if not self.provenance_enabled:
            return
        
        logger.debug(f"Recording model evaluation: {model_id} on {test_data_id}")
        
        # Generate evaluation ID
        evaluation_id = f"evaluation_{uuid.uuid4()}"
        
        # Store evaluation information
        self.operations[evaluation_id] = {
            "operation_id": evaluation_id,
            "operation_name": "model_evaluation",
            "model_id": model_id,
            "test_data_id": test_data_id,
            "evaluation_results": evaluation_results,
            "timestamp": datetime.datetime.now().isoformat(),
            "run_id": self.run_id
        }
        
        # Add evaluation results to model in lineage graph
        if model_id in self.lineage_graph:
            self.lineage_graph[model_id]["evaluation"] = {
                "evaluation_id": evaluation_id,
                "test_data_id": test_data_id,
                "results": evaluation_results
            }
        
        if test_data_id in self.lineage_graph:
            self.lineage_graph[test_data_id]["outputs"].append({
                "id": model_id,
                "via_evaluation": evaluation_id,
                "role": "test_data"
            })
        
        # Save to disk
        if self.provenance_db_path:
            self._save_provenance_record("operations", evaluation_id, self.operations[evaluation_id])
    
    def _save_provenance_record(self, record_type: str, record_id: str, record_data: Dict[str, Any]) -> None:
        """Save a provenance record to disk."""
        # Create record type directory if it doesn't exist
        record_type_dir = os.path.join(self.provenance_db_path, record_type)
        os.makedirs(record_type_dir, exist_ok=True)
        
        # Save record to disk
        record_path = os.path.join(record_type_dir, f"{record_id}.json")
        with open(record_path, 'w') as f:
            json.dump(record_data, f, indent=2, default=str)
    
    def save_provenance_graph(self, output_path: Optional[str] = None) -> str:
        """
        Save the complete provenance graph to disk.
        
        Args:
            output_path: Optional path to save the graph (defaults to provenance DB path)
            
        Returns:
            Path to the saved graph file
        """
        if not self.provenance_enabled:
            logger.warning("Provenance tracking is disabled, no graph to save")
            return ""
        
        # If output_path is not provided, use provenance DB path
        if output_path is None:
            output_path = os.path.join(self.provenance_db_path, f"lineage_graph_{self.run_id}.json")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Compile complete provenance graph
        complete_graph = {
            "run_id": self.run_id,
            "start_time": self.start_time,
            "end_time": datetime.datetime.now().isoformat(),
            "lineage_graph": self.lineage_graph,
            "data_sources": self.data_sources,
            "transformations": self.transformations,
            "data_splits": self.data_splits,
            "operations": self.operations
        }
        
        # Save to disk
        with open(output_path, 'w') as f:
            json.dump(complete_graph, f, indent=2, default=str)
        
        logger.info(f"Provenance graph saved to {output_path}")
        return output_path
    
    def get_data_provenance(self) -> Dict[str, Any]:
        """Get the current data provenance information."""
        if not self.provenance_enabled:
            return {}
        
        return {
            "run_id": self.run_id,
            "start_time": self.start_time,
            "data_sources": list(self.data_sources.keys()),
            "transformations": list(self.transformations.keys()),
            "data_splits": list(self.data_splits.keys())
        }
    
    def get_lineage_for_data(self, data_id: str) -> Dict[str, Any]:
        """
        Get the complete lineage for a data entity.
        
        Args:
            data_id: ID of the data entity
            
        Returns:
            Complete lineage information for the data entity
        """
        if not self.provenance_enabled or data_id not in self.lineage_graph:
            return {}
        
        # Start with the data entity
        lineage = {
            "data_id": data_id,
            "ancestors": [],
            "descendants": []
        }
        
        # Recursively trace ancestors
        self._trace_ancestors(data_id, lineage["ancestors"])
        
        # Recursively trace descendants
        self._trace_descendants(data_id, lineage["descendants"])
        
        return lineage
    
    def _trace_ancestors(self, data_id: str, ancestors_list: List[Dict[str, Any]]) -> None:
        """Recursively trace ancestors of a data entity."""
        if data_id not in self.lineage_graph:
            return
        
        entity = self.lineage_graph[data_id]
        
        for input_info in entity.get("inputs", []):
            input_id = input_info.get("id")
            
            if input_id in self.lineage_graph:
                ancestor_info = {
                    "data_id": input_id,
                    "relationship": input_info,
                    "entity_type": self.lineage_graph[input_id].get("type"),
                    "ancestors": []
                }
                
                ancestors_list.append(ancestor_info)
                
                # Recursively trace ancestors of this ancestor
                self._trace_ancestors(input_id, ancestor_info["ancestors"])
    
    def _trace_descendants(self, data_id: str, descendants_list: List[Dict[str, Any]]) -> None:
        """Recursively trace descendants of a data entity."""
        if data_id not in self.lineage_graph:
            return
        
        entity = self.lineage_graph[data_id]
        
        for output_info in entity.get("outputs", []):
            output_id = output_info.get("id")
            
            if output_id in self.lineage_graph:
                descendant_info = {
                    "data_id": output_id,
                    "relationship": output_info,
                    "entity_type": self.lineage_graph[output_id].get("type"),
                    "descendants": []
                }
                
                descendants_list.append(descendant_info)
                
                # Recursively trace descendants of this descendant
                self._trace_descendants(output_id, descendant_info["descendants"])
    
    def visualize_provenance_graph(self, output_path: Optional[str] = None) -> str:
        """
        Generate a visualization of the provenance graph.
        
        Args:
            output_path: Optional path to save the visualization
            
        Returns:
            Path to the saved visualization file
        """
        if not self.provenance_enabled:
            logger.warning("Provenance tracking is disabled, no graph to visualize")
            return ""
        
        # Check if graphviz is available
        try:
            import graphviz
        except ImportError:
            logger.warning("Graphviz not available, cannot generate visualization")
            return ""
        
        # If output_path is not provided, use provenance DB path
        if output_path is None:
            output_path = os.path.join(self.provenance_db_path, f"lineage_graph_{self.run_id}.svg")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create a directed graph
        dot = graphviz.Digraph(comment='Data Provenance Graph', format='svg')
        
        # Define node styles
        node_styles = {
            "data_source": {"shape": "cylinder", "color": "blue", "style": "filled", "fillcolor": "lightblue"},
            "transformed_data": {"shape": "cylinder", "color": "green", "style": "filled", "fillcolor": "lightgreen"},
            "train_data": {"shape": "cylinder", "color": "orange", "style": "filled", "fillcolor": "lightsalmon"},
            "validation_data": {"shape": "cylinder", "color": "purple", "style": "filled", "fillcolor": "plum"},
            "test_data": {"shape": "cylinder", "color": "red", "style": "filled", "fillcolor": "lightpink"},
            "trained_model": {"shape": "box", "color": "black", "style": "filled", "fillcolor": "gray90"}
        }
        
        # Add nodes for all entities in the lineage graph
        for entity_id, entity_info in self.lineage_graph.items():
            entity_type = entity_info.get("type", "unknown")
            label = f"{entity_type}\\n{entity_id[:8]}..."
            
            # Get node style based on entity type
            style = node_styles.get(entity_type, {"shape": "ellipse"})
            
            # Add node
            dot.node(entity_id, label, **style)
        
        # Add edges for relationships
        for entity_id, entity_info in self.lineage_graph.items():
            # Add edges for inputs
            for input_info in entity_info.get("inputs", []):
                input_id = input_info.get("id")
                relationship = next((k for k in input_info.keys() if k.startswith("via_")), None)
                
                if input_id in self.lineage_graph and relationship:
                    # Label the edge with the relationship type
                    label = relationship.replace("via_", "")
                    dot.edge(input_id, entity_id, label=label)
        
        # Render the graph
        dot.render(output_path, cleanup=True)
        
        logger.info(f"Provenance graph visualization saved to {output_path}")
        return output_path
    

