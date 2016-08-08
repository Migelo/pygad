GSL_HOME	:= ${GSL_HOME} /opt/local

CC 		= g++
CFLAGS 	= -Wall -std=c++11
CFLAGS 	+= -Ofast -DNDEBUG -fopenmp
#CFLAGS 	+= -Wno-unknown-pragmas -O2 -DDEBUG
IPATH	= $(addprefix -I, $(addsuffix /include, $(GSL_HOME)))
LPATH	= $(addprefix -L, $(addsuffix /lib, $(GSL_HOME)))
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

