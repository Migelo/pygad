GSL_INCLUDE	= ${GSL_HOME}/include -I/opt/local/include
GSL_LIB		= ${GSL_HOME}/lib -I/opt/local/lib

CC 		= g++
CFLAGS 	= -Wall -std=c++11 -Ofast -DNDEBUG -fopenmp
#CFLAGS 	= -Wall -std=c++11 -O2 -fopenmp
#CFLAGS 	= -Wall -std=c++11 -O0 -DDEBUG
IPATH	= $(addprefix -I,$(GSL_INCLUDE))
LPATH	= $(addprefix -L,$(GSL_LIB))
LDFLAGS = -lm -fopenmp -lgsl -lgslcblas

LIB		= cpygad.so
SRCDIR	= src
INCLDIR	= include
BUILDIR	= build


# derived variables
SRC		= $(wildcard $(SRCDIR)/*.cpp)
OBJ 	= $(addprefix $(BUILDIR)/,$(notdir $(SRC:%.cpp=%.o)))
HEADERS = $(wildcard $(INCLDIR)/*.hpp)


.PHONY:	all clean
.SECONDARY: main-build

all: $(LIB)

# also let depend on headers to enshure proper rebuilds
$(OBJ) : $(BUILDIR)/%.o : $(SRCDIR)/%.cpp $(HEADERS) Makefile
	@mkdir -p $(BUILDIR)
	$(CC) -fPIC $(CFLAGS) $(IPATH) -I./$(INCLDIR) -c $< -o $@

$(LIB): $(OBJ) Makefile
	$(CC) -fPIC -shared $(LPATH) $(OBJ) $(LDFLAGS) -o $@

clean:
	$(RM) -r $(BUILDIR)
	$(RM) $(LIB)

