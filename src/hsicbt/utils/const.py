from .misc import get_current_timestamp

TTYPE_STANDARD = "backprop"
TTYPE_HSICTRAIN = "hsictrain"
TTYPE_FORMAT = "format"
TTYPE_UNFORMAT = "unformat"
TTYPE_PLOT = "plot"

FONTSIZE_TITLE = 60
FONTSIZE_XLABEL = 50
FONTSIZE_YLABEL = 50
FONTSIZE_XTICKS = 40
FONTSIZE_YTICKS = 40
FONTSIZE_LEDEND = 40
FONTSIZE_FOOTNOTE = 10

DEBUG_MODE = 0

# A single timestamp shared by every asset written during one process run.
TIMESTAMP_CODE = get_current_timestamp()
