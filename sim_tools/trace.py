"""
Simple functionality aiming to enhanced a users a
ability to trace and debug simulation models.
"""

from abc import ABC
from rich.console import Console

DEFAULT_DEBUG = False

CONFIG_ERROR = ("Your trace has not been initialised. " 
                "Call super__init__(debug=True) in class initialiser" 
                "or omit debug for default of no trace.")


## single rich console - module level.
_console = Console()

class Traceable(ABC):
    '''Provides basic trace functionality for a process to subclass
    
    Abstract base class Traceable
    
    Subclasses must call 
    
    super().__init__(debug=True) in their __init__() method to 
    initialise trace.
    
    Subclasses inherit the following methods:

    trace() - use this function print out a traceable event

    _trace_config(): use this function to return a dict containing
    the trace configuration for the class.
    '''
    def __init__(self, debug=DEFAULT_DEBUG):
        self.debug = debug
        self._config = self._default_config()
    
    def _default_config(self):
        """Returns a default trace configuration"""
        config = {
            "name":None, 
            "name_colour":"bold blue", 
            "time_colour":'bold blue', 
            "time_dp":2,
            "message_colour":'black',
            "tracked":None
        }
        return config
        
    
    def _trace_config(self):
        config = {
            "name":None, 
            "name_colour":"bold blue", 
            "time_colour":'bold blue', 
            "time_dp":2,
            "message_colour":'black',
            "tracked":None
        }
        return config
    
    
    def trace(self, time, msg=None, process_id=None):
        '''
        Display a trace of an event
        '''
        
        if not hasattr(self, '_config'):
            raise AttributeError(CONFIG_ERROR)
        
        # if in debug mode
        if self.debug:
            
            # check for override to default configs
            process_config = self._trace_config()
            self._config.update(process_config)
            
            # conditional logic to limit tracking to specific processes/entities
            if self._config['tracked'] is None or process_id in self._config['tracked']:

                # display and format time stamp
                out = f"[{self._config['time_colour']}][{time:.{self._config['time_dp']}f}]:[/{self._config['time_colour']}]"
                
                # if provided display and format a process ID 
                if self._config['name'] is not None and process_id is not None:
                    out += f"[{self._config['name_colour']}]<{self._config['name']} {process_id}>: [/{self._config['name_colour']}]"

                # format traced event message
                out += f"[{self._config['message_colour']}]{msg}[/{self._config['message_colour']}]"

                # print to rich console
                _console.print(out)
        