"""
Pipeline Processor Module - AstroBin Upload Utility v2.1.0

This module implements the Pipeline Design Pattern, which is the architectural 
core of the v2.0 application. By decoupling complex metadata transformations 
into discrete 'Steps', we ensure that the logic is modular, testable, and 
easy to extend.

Each Step in the pipeline operates on a shared 'SessionState' object, 
modifying it in place or replacing its internal DataFrames as it flows 
through the execution sequence.
"""

import logging
import os
from typing import List, Protocol
from models import SessionState

class PipelineStep(Protocol):
    """
    Interface for a single transformation step in the pipeline.
    
    Any class that implements an 'execute' method accepting and returning 
    a SessionState object is a valid PipelineStep. This protocol-based 
    approach allows for loose coupling between the processor and its logic.
    """
    def execute(self, state: SessionState) -> SessionState:
        """
        Runs the specific transformation logic for this step.
        
        Args:
            state (SessionState): The current state of the session.
            
        Returns:
            SessionState: The updated session state.
        """
        ...

class PipelineProcessor:
    """
    Orchestrator responsible for chaining and executing pipeline steps.
    
    The processor maintains an ordered list of steps and ensures that 
    the SessionState flows through them sequentially, respecting 
    data dependencies (e.g., normalization must happen before aggregation).
    """
    def __init__(self, logger: logging.Logger):
        """
        Initializes the processor with a system logger.

        Args:
            logger (logging.Logger): The active application logger.
        """
        self.logger = logger
        self.steps: List[PipelineStep] = []

    def add_step(self, step: PipelineStep):
        """
        Registers a new step to be executed in the sequence.
        
        Steps are executed in the order they are added.

        Args:
            step (PipelineStep): An instance of a class implementing the PipelineStep protocol.
        """
        self.logger.debug(f"Registered pipeline step: {step.__class__.__name__}")
        self.steps.append(step)

    def run(self, state: SessionState, debug: bool = False, output_dir: str = None) -> SessionState:
        """
        Executes all registered steps sequentially on the provided state.
        
        This is the primary execution loop of the application's processing engine.

        Args:
            state (SessionState): The initial data and configuration state.
            debug (bool): If True, preserve intermediate dataframes for troubleshooting.
            output_dir (str): Directory where debug artifacts should be saved.
            
        Returns:
            SessionState: The fully transformed and processed data state.
        """
        self.logger.info("Processing state initialized")
        self.logger.info(f"Pipeline execution started: {len(self.steps)} steps registered.")
        
        # Sequentially flow the state through each registered transformation
        for i, step in enumerate(self.steps, 1):
            step_name = step.__class__.__name__
            self.logger.debug(f"Executing step: {step_name}")
            
            try:
                state = step.execute(state)
                
                # Sequential Debug CSV Exports (On Success)
                if debug and output_dir:
                    self._dump_debug_csv(state, i, step_name, output_dir)

            except Exception as e:
                # Always attempt to dump a diagnostic CSV on failure for troubleshooting,
                # even if --debug was not specified.
                if output_dir:
                    try:
                        self._dump_debug_csv(state, i, step_name, output_dir, suffix="_CRASH_DIAGNOSTIC")
                        self.logger.info(f"Emergency diagnostic state saved to {output_dir}")
                    except: 
                        pass # Prevent diagnostic failure from masking the real error

                error_msg = f"CRITICAL FAILURE in [{step_name}]: {str(e)}"
                self.logger.error(error_msg)
                
                # Always log the full traceback to the log file for troubleshooting
                self.logger.exception(e)
                
                # Re-raise to stop the pipeline
                raise
            
        self.logger.info("Pipeline execution completed successfully.")
        return state

    def _dump_debug_csv(self, state: SessionState, step_index: int, step_name: str, output_dir: str, suffix: str = ""):
        """Helper to export the current session state to a CSV for debugging."""
        try:
            csv_filename = f"debug_step_{step_index:02d}_{step_name}{suffix}.csv"
            csv_path = os.path.join(output_dir, csv_filename)
            
            # Priority 1: Save aggregated_df if we are at/past AggregationStep
            if step_name == "AggregationStep" and not state.aggregated_df.empty:
                state.aggregated_df.to_csv(csv_path, index=False)
                self.logger.debug(f"Saved debug state (aggregated) to {csv_path}")
            
            # Priority 2: Save processed_df if it exists and is not empty
            elif not state.processed_df.empty:
                state.processed_df.to_csv(csv_path, index=False)
                self.logger.debug(f"Saved debug state to {csv_path}")
                
            # Priority 3: If very first step failed, save raw_df if available
            elif step_index == 1 and not state.raw_df.empty:
                state.raw_df.to_csv(csv_path, index=False)
                self.logger.debug(f"Saved debug state (raw) to {csv_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to save debug CSV for {step_name}: {str(e)}")
