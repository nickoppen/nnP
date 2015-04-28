#define CORECOUNT 16
#define LOCAL_MEM_ADDRESS_BASE(gid) (((32 + (gid / 4)) << 26) | ((8 + (gid % 4)) << 20))

#define core00 0x80800000
#define core10 0x80900000
#define core20 0x80a00000
#define core30 0x80b00000
#define core01 0x84800000
#define core11 0x84900000
#define core21 0x84a00000
#define core31 0x84b00000
#define core02 0x88800000
#define core12 0x88900000
#define core22 0x88a00000
#define core32 0x88b00000
#define core03 0x8c800000
#define core13 0x8c900000
#define core23 0x8ca00000
#define core33 0x8cb00000

#define NEIGHBOUR_LOC(CORE, STRUCTURE, INDEX, SIZEOFTYPE) (CORE + ((unsigned int)STRUCTURE) + (INDEX * SIZEOFTYPE))
#define NEIGHBOR_LOC(CORE, STRUCTURE, INDEX, SIZEOFTYPE) (CORE + ((unsigned int)STRUCTURE) + (INDEX * SIZEOFTYPE))      /// for those who use the "alternative" spelling

/// usage:
//
///         *(int*)NEIGHBOUR_LOC(NEXT, vLocal, i, sizeof(int)) = vLocal[i];
//
/// where the contents of vLocal are ints
