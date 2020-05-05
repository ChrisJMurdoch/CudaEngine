
#pragma once

#include "..\..\include\models\sModel.hpp"

/** Rewritable, fixed size mesh model */
class DModel : public SModel
{
public:
    /** Chains to protected ctor with GL_STREAM_DRAW */
    DModel(float *vertexData, int nVertices);
    void setVertexData(float *vertexData, int nVertices);
};
