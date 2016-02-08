CC 		= g++
CFLAGS 	= -Wall -std=c++14 -Ofast -DNDEBUG -fopenmp
#CFLAGS 	= -Wall -std=c++14 -O2 -fopenmp
#CFLAGS 	= -Wall -std=c++14 -O0 -DDEBUG
LDFLAGS = -L/opt/local/lib -lm -fopenmp -lgsl -lgslcblas

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
	$(CC) $(CFLAGS) -I./$(INCLDIR) -c $< -o $@

$(LIB): $(OBJ) Makefile
	$(CC) -shared $(LDFLAGS) $(OBJ) -o $@

clean:
	$(RM) -r $(BUILDIR)
	$(RM) $(LIB)

