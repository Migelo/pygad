CC 		= g++
CFLAGS 	= -Wall -std=c++14 -Ofast -fopenmp
LDFLAGS = -lm

LIB		= cpygad.so
SRCDIR	= src
INCLDIR	= include
BUILDIR	= build


# derived variables
SRC		= $(wildcard $(SRCDIR)/*.cpp)
OBJ 	= $(addprefix $(BUILDIR)/,$(notdir $(SRC:%.cpp=%.o)))
HEADERS = $(wildcard $(INCLDIR)/*.h)


.PHONY:	all clean
.SECONDARY: main-build

all: $(LIB)

# also let depend on headers to enshure proper rebuilds
$(OBJ) : $(BUILDIR)/%.o : $(SRCDIR)/%.cpp $(HEADERS)
	@mkdir -p $(BUILDIR)
	$(CC) $(CFLAGS) -I./$(INCLDIR) -c $< -o $@

$(LIB): $(OBJ)
	$(CC) -shared $(LDFLAGS) $^ -o $@

clean:
	$(RM) -r $(BUILDIR)
	$(RM) $(LIB)

